from ..Qt import QtCore, QtGui, QtWidgets
import math
import sys
import warnings
import numpy as np
from .. import Qt, debug
from .. import functions as fn
from .. import getConfigOption
from .GraphicsObject import GraphicsObject
def _getFillPath(self):
    if self.fillPath is not None:
        return self.fillPath
    path = QtGui.QPainterPath(self.getPath())
    self.fillPath = path
    if self.opts['fillLevel'] == 'enclosed':
        return path
    baseline = self.opts['fillLevel']
    x, y = self.getData()
    lx, rx = x[[0, -1]]
    ly, ry = y[[0, -1]]
    if ry != baseline:
        path.lineTo(rx, baseline)
    path.lineTo(lx, baseline)
    if ly != baseline:
        path.lineTo(lx, ly)
    return path