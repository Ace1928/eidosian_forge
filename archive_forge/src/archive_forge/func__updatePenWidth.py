import numpy as np
from .. import functions as fn
from .. import getConfigOption
from .. import Qt
from ..Qt import QtCore, QtGui
from .GraphicsObject import GraphicsObject
def _updatePenWidth(self, pen):
    no_pen = pen is None or pen.style() == QtCore.Qt.PenStyle.NoPen
    if no_pen:
        return
    idx = pen.isCosmetic()
    self._penWidth[idx] = max(self._penWidth[idx], pen.widthF())