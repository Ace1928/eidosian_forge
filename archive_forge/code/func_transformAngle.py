import operator
import weakref
from collections import OrderedDict
from functools import reduce
from math import hypot
from typing import Optional
from xml.etree.ElementTree import Element
from .. import functions as fn
from ..GraphicsScene import GraphicsScene
from ..Point import Point
from ..Qt import QtCore, QtWidgets, isQObjectAlive
def transformAngle(self, relativeItem=None):
    """Return the rotation produced by this item's transform (this assumes there is no shear in the transform)
        If relativeItem is given, then the angle is determined relative to that item.
        """
    if relativeItem is None:
        relativeItem = self.parentItem()
    tr = self.itemTransform(relativeItem)[0]
    vec = tr.map(QtCore.QLineF(0, 0, 1, 0))
    return vec.angleTo(QtCore.QLineF(vec.p1(), vec.p1() + QtCore.QPointF(1, 0)))