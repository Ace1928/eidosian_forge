import warnings
from collections.abc import Callable
import numpy
from .. import colormap
from .. import debug as debug
from .. import functions as fn
from .. import functions_qimage
from .. import getConfigOption
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets
from ..util.cupy_helper import getCupy
from .GraphicsObject import GraphicsObject
def _computeDownsampleFactors(self):
    o = self.mapToDevice(QtCore.QPointF(0, 0))
    x = self.mapToDevice(QtCore.QPointF(1, 0))
    y = self.mapToDevice(QtCore.QPointF(0, 1))
    if o is None:
        return (None, None)
    w = Point(x - o).length()
    h = Point(y - o).length()
    if w == 0 or h == 0:
        return (None, None)
    return (max(1, int(1.0 / w)), max(1, int(1.0 / h)))