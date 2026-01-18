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
def tickClicked(self, tick, ev):
    if ev.button() == QtCore.Qt.MouseButton.LeftButton:
        self.raiseColorDialog(tick)
    elif ev.button() == QtCore.Qt.MouseButton.RightButton:
        self.raiseTickContextMenu(tick, ev)