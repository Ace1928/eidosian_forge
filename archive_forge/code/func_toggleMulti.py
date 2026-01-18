import weakref
from .. import functions as fn
from ..graphicsItems.GraphicsObject import GraphicsObject
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets
def toggleMulti(self):
    multi = self.menu.multiAct.isChecked()
    self.term.setMultiValue(multi)