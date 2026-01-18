import weakref
from .. import functions as fn
from ..graphicsItems.GraphicsObject import GraphicsObject
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets
def valueIsAcceptable(self):
    """Returns True->acceptable  None->unknown  False->Unacceptable"""
    return self.valueOk