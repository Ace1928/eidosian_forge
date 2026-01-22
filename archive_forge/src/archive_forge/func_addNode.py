import importlib
import os
from collections import OrderedDict
from numpy import ndarray
from .. import DataTreeWidget, FileDialog
from .. import configfile as configfile
from .. import dockarea as dockarea
from .. import functions as fn
from ..debug import printExc
from ..graphicsItems.GraphicsObject import GraphicsObject
from ..Qt import QtCore, QtWidgets
from . import FlowchartCtrlTemplate_generic as FlowchartCtrlTemplate
from . import FlowchartGraphicsView
from .library import LIBRARY
from .Node import Node
from .Terminal import Terminal
def addNode(self, node):
    ctrl = node.ctrlWidget()
    item = QtWidgets.QTreeWidgetItem([node.name(), '', ''])
    self.ui.ctrlList.addTopLevelItem(item)
    byp = QtWidgets.QPushButton('X')
    byp.setCheckable(True)
    byp.setFixedWidth(20)
    item.bypassBtn = byp
    self.ui.ctrlList.setItemWidget(item, 1, byp)
    byp.node = node
    node.bypassButton = byp
    byp.setChecked(node.isBypassed())
    byp.clicked.connect(self.bypassClicked)
    if ctrl is not None:
        item2 = QtWidgets.QTreeWidgetItem()
        item.addChild(item2)
        self.ui.ctrlList.setItemWidget(item2, 0, ctrl)
    self.items[node] = item