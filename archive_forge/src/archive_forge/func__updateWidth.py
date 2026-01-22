import weakref
from math import ceil, floor, isfinite, log10, sqrt, frexp, floor
import numpy as np
from .. import debug as debug
from .. import functions as fn
from .. import getConfigOption
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets
from .GraphicsWidget import GraphicsWidget
def _updateWidth(self):
    if not self.isVisible():
        w = 0
    elif self.fixedWidth is None:
        if not self.style['showValues']:
            w = 0
        elif self.style['autoExpandTextSpace']:
            w = self.textWidth
        else:
            w = self.style['tickTextWidth']
        w += self.style['tickTextOffset'][0] if self.style['showValues'] else 0
        w += max(0, self.style['tickLength'])
        if self.label.isVisible():
            w += self.label.boundingRect().height() * 0.8
    else:
        w = self.fixedWidth
    self.setMaximumWidth(w)
    self.setMinimumWidth(w)
    self.picture = None