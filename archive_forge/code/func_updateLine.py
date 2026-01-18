import weakref
from .. import functions as fn
from ..graphicsItems.GraphicsObject import GraphicsObject
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets
def updateLine(self):
    start = Point(self.source.connectPoint())
    if isinstance(self.target, TerminalGraphicsItem):
        stop = Point(self.target.connectPoint())
    elif isinstance(self.target, QtCore.QPointF):
        stop = Point(self.target)
    else:
        return
    self.prepareGeometryChange()
    self.path = self.generatePath(start, stop)
    self.shapePath = None
    self.update()