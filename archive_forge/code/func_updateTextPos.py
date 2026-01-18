from math import atan2, degrees
from .. import functions as fn
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets
from .GraphicsObject import GraphicsObject
def updateTextPos(self):
    r = self.textItem.boundingRect()
    tl = self.textItem.mapToParent(r.topLeft())
    br = self.textItem.mapToParent(r.bottomRight())
    offset = (br - tl) * self.anchor
    self.textItem.setPos(-offset)