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
def viewRect(self):
    """Return a QRectF bounding the region visible within the ViewBox"""
    try:
        vr0 = self.state['viewRange'][0]
        vr1 = self.state['viewRange'][1]
        return QtCore.QRectF(vr0[0], vr1[0], vr0[1] - vr0[0], vr1[1] - vr1[0])
    except:
        print('make qrectf failed:', self.state['viewRange'])
        raise