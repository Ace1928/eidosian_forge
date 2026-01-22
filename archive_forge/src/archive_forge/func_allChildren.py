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
def allChildren(self, item=None):
    """Return a list of all children and grandchildren of this ViewBox"""
    if item is None:
        item = self.childGroup
    children = [item]
    for ch in item.childItems():
        children.extend(self.allChildren(ch))
    return children