from ..graphicsItems.ViewBox import ViewBox
from ..Qt import QtCore, QtGui, QtWidgets
from ..widgets.GraphicsView import GraphicsView
class FlowchartGraphicsView(GraphicsView):
    sigHoverOver = QtCore.Signal(object)
    sigClicked = QtCore.Signal(object)

    def __init__(self, widget, *args):
        GraphicsView.__init__(self, *args, useOpenGL=False)
        self._vb = FlowchartViewBox(widget, lockAspect=True, invertY=True)
        self.setCentralItem(self._vb)
        self.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)

    def viewBox(self):
        return self._vb