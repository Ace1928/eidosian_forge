import gc
import importlib
import weakref
import warnings
from ..graphicsItems.GridItem import GridItem
from ..graphicsItems.ROI import ROI
from ..graphicsItems.ViewBox import ViewBox
from ..Qt import QT_LIB, QtCore, QtGui, QtWidgets
from . import CanvasTemplate_generic as ui_template
from .CanvasItem import CanvasItem, GroupCanvasItem
from .CanvasManager import CanvasManager
def treeItemMoved(self, item, parent, index):
    if parent is self.itemList.invisibleRootItem():
        item.canvasItem().setParentItem(self.view.childGroup)
    else:
        item.canvasItem().setParentItem(parent.canvasItem())
    siblings = [parent.child(i).canvasItem() for i in range(parent.childCount())]
    zvals = [i.zValue() for i in siblings]
    zvals.sort(reverse=True)
    for i in range(len(siblings)):
        item = siblings[i]
        item.setZValue(zvals[i])