# -*- coding:UTF-8 -*-
from sklearn.linear_model import LogisticRegression
import numpy as np
import random

#矩阵与数组  矩阵是二维的所以与数组相乘，数组必须是二维的
n = np.mat([4,4,6])
m = np.array([[4,4,6]])
m2 =m.reshape(3,1)
m1=m2*n         #与矩阵相乘一样，得满足矩阵条件,得到的结果还是一个矩阵
m3=n*m2         #与矩阵相乘一样，得满足矩阵条件,得到的结果还是一个矩阵


#list与矩阵,list与矩阵不能相乘！！！
m =[1,2,3]
n = np.mat([4,4,6])
m1 = m *n     #这里会出现error



#list与数组
m =[1,2,3]
n = np.array([[4,4,6]])   #二维数组
m0 = m *n                 #相当于两个行向量相乘，为对应元素相乘 m0=array([[ 4,  8, 18]])   >>> m0.shape = (1L, 3L)    type(m0)=<type 'numpy.ndarray'>
n1 = n.ravel()            #变成1维数组  n1.shape = (3L,)
m1 = m *n1                #相当于两个行向量相乘，为对应元素相乘 m0=array([[ 4,  8, 18]])   >>> m0.shape = (1L, 3L)    type(m0)=<type 'numpy.ndarray'>
n2 = n1.reshape(3,-1)     #n2.shape = (3L, 1L)
m1=m*n2                   # array([[ 4,  8, 12],
                          #      [ 4,  8, 12],
                          #     [ 6, 12, 18]])    m1.shape=(3L, 3L)  相当于是将矩阵的列扩展开来

#矩阵的相关操作matrix只能是二维的
b=np.mat([1,2])         #b.shape=(1L, 2L)  b[0] = matrix([[1, 2]])   b[0].shape = (1L, 2L)  
b1=np.mat([[1,2]])      #b1.shape=(1L, 2L) b1[0] = matrix([[1, 2]])   b1[0].shape = (1L, 2L)
b2=np.mat([[1,2],[3,4],[5,6]])   #b2.shape= (3L, 2L)     b2[0]=matrix([[1, 2]])  b2[0].shape=(1L, 2L)
                                 #索引矩阵某一行b[0]   某一列  b2[:,1]   某个值 b2[1,1]
b3 = np.mat([[1,2],[3,4]])
b4 = b2 *b3                     #矩阵乘法，与数组的.dot作用一样
b5 = np.mat([3,4]) 
b6 = np.multiply(b3,b5)         #对应元素相乘

#数组的相关操作
a = np.array([[4,4,6]])             #有两个方括号，是二维数组    >>> a.shape = (1L, 3L),二维的索引只能a[0][0],a[0]索引出来的是一维数组  >>> a[0].shape = (3L,)
a1 = np.array([[4,4,6],[4,5,6]])    #>>> a.shape = (2L, 3L)
a2= np.array([1,2,3])               #只有一个括号，是一维数组  >>> a1.shape = (3L,)
a3 = np.array([4,4,6])
#形状改变
a4 = a1.ravel()                     #将二维编程一维数组    >>> a4 = array([4, 4, 6, 4, 5, 6])   >>> a4.shape=(6L,)
a5 = a1.reshape(3,-1)               #3代表行数，将数组划分为3行，列数根据数组总个数来划分  a5 = array([[4, 4],
                                                                                #                   [6, 4],
                                                                                #                   [5, 6]])
#数组运算
a6 = a1**2                         # a5=array([[16, 16, 36],
                                   #          [16, 25, 36]])   对应元素平方
a7 = a2*a3                         #一维数组相乘，等于对应元素相乘，结果还是数组      a6=array([ 4,  8, 18])
a8 = ([[1,2],[3,4],[5,6]])
a9 = a1.dot(a8)                    #数组相乘，多维数组相乘，与矩阵运算相同，只能用.dot，用*时两个数组的行数与列数必须都相同    a9=array([[46, 60],
                                                              #                                                                  [49, 64]])  
a10 = ([[1,2,3],[4,5,6]])
a11 = a10 * a1                     #  >>> a11=array([[ 4,  8, 18],
                                   #                 [16, 25, 36]])         数组相乘，对应元素相乘，两个数组行数列数必须相同  
                                   


