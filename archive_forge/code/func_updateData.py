import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
def updateData():
    yd, xd = rand(10000)
    p1.setData(y=yd, x=xd)