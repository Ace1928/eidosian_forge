import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
def updateRoiPlot(roi, data=None):
    if data is None:
        data = roi.getArrayRegion(im1.image, img=im1)
    if data is not None:
        roi.curve.setData(data.mean(axis=1))