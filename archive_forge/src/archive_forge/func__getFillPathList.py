from ..Qt import QtCore, QtGui, QtWidgets
import math
import sys
import warnings
import numpy as np
from .. import Qt, debug
from .. import functions as fn
from .. import getConfigOption
from .GraphicsObject import GraphicsObject
def _getFillPathList(self, widget):
    if self._fillPathList is not None:
        return self._fillPathList
    x, y = self.getData()
    if self.opts['stepMode']:
        x, y = self._generateStepModeData(self.opts['stepMode'], x, y, baseline=None)
    if not self.opts['skipFiniteCheck']:
        mask = np.isfinite(x) & np.isfinite(y)
        if not mask.all():
            x = x[mask]
            y = y[mask]
    if len(x) < 2:
        return []
    chunksize = 50 if not isinstance(widget, QtWidgets.QOpenGLWidget) else 5000
    paths = self._fillPathList = []
    offset = 0
    xybuf = np.empty((chunksize + 3, 2))
    baseline = self.opts['fillLevel']
    while offset < len(x) - 1:
        subx = x[offset:offset + chunksize]
        suby = y[offset:offset + chunksize]
        size = len(subx)
        xyview = xybuf[:size + 3]
        xyview[:-3, 0] = subx
        xyview[:-3, 1] = suby
        xyview[-3:, 0] = subx[[-1, 0, 0]]
        xyview[-3:, 1] = [baseline, baseline, suby[0]]
        offset += size - 1
        path = fn._arrayToQPath_all(xyview[:, 0], xyview[:, 1], finiteCheck=False)
        paths.append(path)
    return paths