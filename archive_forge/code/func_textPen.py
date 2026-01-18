import weakref
from math import ceil, floor, isfinite, log10, sqrt, frexp, floor
import numpy as np
from .. import debug as debug
from .. import functions as fn
from .. import getConfigOption
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets
from .GraphicsWidget import GraphicsWidget
def textPen(self):
    if self._textPen is None:
        return fn.mkPen(getConfigOption('foreground'))
    return fn.mkPen(self._textPen)