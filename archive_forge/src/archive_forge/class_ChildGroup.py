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
class ChildGroup(ItemGroup):

    def __init__(self, parent):
        ItemGroup.__init__(self, parent)
        self.itemsChangedListeners = WeakList()
        self._GraphicsObject__inform_view_on_change = False

    def itemChange(self, change, value):
        ret = ItemGroup.itemChange(self, change, value)
        if change in [self.GraphicsItemChange.ItemChildAddedChange, self.GraphicsItemChange.ItemChildRemovedChange]:
            try:
                itemsChangedListeners = self.itemsChangedListeners
            except AttributeError:
                pass
            else:
                for listener in itemsChangedListeners:
                    listener.itemsChanged()
        return ret