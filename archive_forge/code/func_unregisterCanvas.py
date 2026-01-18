from ..Qt import QtCore, QtWidgets
import weakref
def unregisterCanvas(self, name):
    c = self.canvases[name]
    del self.canvases[name]
    self.sigCanvasListChanged.emit()