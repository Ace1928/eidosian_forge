from time import perf_counter
import numpy as np
import pyqtgraph as pg
def update1():
    global data1, ptr1
    data1[:-1] = data1[1:]
    data1[-1] = np.random.normal()
    curve1.setData(data1)
    ptr1 += 1
    curve2.setData(data1)
    curve2.setPos(ptr1, 0)