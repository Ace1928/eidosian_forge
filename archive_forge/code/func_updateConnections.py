import weakref
from .. import functions as fn
from ..graphicsItems.GraphicsObject import GraphicsObject
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets
def updateConnections(self):
    for t, c in self.term.connections().items():
        c.updateLine()