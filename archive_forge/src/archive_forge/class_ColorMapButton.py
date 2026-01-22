import importlib.util
import re
import numpy as np
from .. import colormap
from .. import functions as fn
from ..graphicsItems.GradientPresets import Gradients
from ..Qt import QtCore, QtGui, QtWidgets
class ColorMapButton(ColorMapDisplayMixin, QtWidgets.QWidget):
    sigColorMapChanged = QtCore.Signal(object)

    def __init__(self):
        QtWidgets.QWidget.__init__(self)
        ColorMapDisplayMixin.__init__(self, orientation='horizontal')

    def colorMapChanged(self):
        cmap = self.colorMap()
        self.sigColorMapChanged.emit(cmap)
        self.update()

    def paintEvent(self, evt):
        painter = QtGui.QPainter(self)
        self.paintColorMap(painter, self.contentsRect())
        painter.end()

    def mouseReleaseEvent(self, evt):
        if evt.button() != QtCore.Qt.MouseButton.LeftButton:
            return
        pos = self.mapToGlobal(self.pos())
        pos.setY(pos.y() + self.height())
        self.getMenu().popup(pos)