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
def updateGradient(self):
    self.gradient = self.getGradient()
    self.gradRect.setBrush(QtGui.QBrush(self.gradient))
    self.sigGradientChanged.emit(self)