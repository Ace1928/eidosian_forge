import numpy as np
from ...graphicsItems.LinearRegionItem import LinearRegionItem
from ...Qt import QtCore, QtWidgets
from ...widgets.TreeWidget import TreeWidget
from ..Node import Node
from . import functions
from .common import CtrlNode
class AsType(CtrlNode):
    """Convert an array to a different dtype.
    """
    nodeName = 'AsType'
    uiTemplate = [('dtype', 'combo', {'values': ['float', 'int', 'float32', 'float64', 'float128', 'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64'], 'index': 0})]

    def processData(self, data):
        s = self.stateGroup.state()
        return data.astype(s['dtype'])