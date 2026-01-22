from ..Qt import QtCore, QtGui, QtWidgets
import math
import sys
import warnings
import numpy as np
from .. import Qt, debug
from .. import functions as fn
from .. import getConfigOption
from .GraphicsObject import GraphicsObject
def _shouldUseDrawLineSegments(self, pen):
    mode = self.opts['segmentedLineMode']
    if mode in ('on',):
        return True
    if mode in ('off',):
        return False
    return pen.widthF() > 1.0 and pen.style() == QtCore.Qt.PenStyle.SolidLine and pen.isSolid() and (pen.color().alphaF() == 1.0) and (not self.opts['antialias'])