import operator
import weakref
import numpy as np
from .. import functions as fn
from .. import colormap
from ..colormap import ColorMap
from ..Qt import QtCore, QtGui, QtWidgets
from ..widgets.SpinBox import SpinBox
from ..widgets.ColorMapButton import ColorMapMenu
from .GraphicsWidget import GraphicsWidget
from .GradientPresets import Gradients
def tickMoved(self, tick, pos):
    newX = min(max(0, pos.x()), self.length)
    pos.setX(newX)
    tick.setPos(pos)
    self.ticks[tick] = float(newX) / self.length
    self.sigTicksChanged.emit(self)