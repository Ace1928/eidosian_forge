from ..graphicsItems.PlotItem import PlotItem
from ..Qt import QtCore, QtWidgets
from .GraphicsView import GraphicsView
def viewRangeChanged(self, view, range):
    self.sigRangeChanged.emit(self, range)