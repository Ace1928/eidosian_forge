from ..Qt import QtCore, QtGui, QtWidgets
import math
import sys
import warnings
import numpy as np
from .. import Qt, debug
from .. import functions as fn
from .. import getConfigOption
from .GraphicsObject import GraphicsObject
def _getClosingSegments(self):
    segments = []
    if self.opts['fillLevel'] == 'enclosed':
        return segments
    baseline = self.opts['fillLevel']
    x, y = self.getData()
    lx, rx = x[[0, -1]]
    ly, ry = y[[0, -1]]
    if ry != baseline:
        segments.append(QtCore.QLineF(rx, ry, rx, baseline))
    segments.append(QtCore.QLineF(rx, baseline, lx, baseline))
    if ly != baseline:
        segments.append(QtCore.QLineF(lx, baseline, lx, ly))
    return segments