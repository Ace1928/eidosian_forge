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
def treeItemChanged(self, item, col):
    try:
        citem = item.canvasItem()
    except AttributeError:
        return
    if item.checkState(0) == QtCore.Qt.CheckState.Checked:
        for i in range(item.childCount()):
            item.child(i).setCheckState(0, QtCore.Qt.CheckState.Checked)
        citem.show()
    else:
        for i in range(item.childCount()):
            item.child(i).setCheckState(0, QtCore.Qt.CheckState.Unchecked)
        citem.hide()