import math
import sys
import weakref
from copy import deepcopy
import numpy as np
from ... import debug as debug
from ... import functions as fn
from ... import getConfigOption
from ...Point import Point
from ...Qt import QtCore, QtGui, QtWidgets, isQObjectAlive, QT_LIB
from ..GraphicsWidget import GraphicsWidget
from ..ItemGroup import ItemGroup
from .ViewBoxMenu import ViewBoxMenu
def updateScaleBox(self, p1, p2):
    r = QtCore.QRectF(p1, p2)
    r = self.childGroup.mapRectFromParent(r)
    self.rbScaleBox.setPos(r.topLeft())
    tr = QtGui.QTransform.fromScale(r.width(), r.height())
    self.rbScaleBox.setTransform(tr)
    self.rbScaleBox.show()