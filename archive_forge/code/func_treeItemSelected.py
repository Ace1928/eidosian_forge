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
def treeItemSelected(self):
    sel = self.selectedItems()
    if len(sel) == 0:
        return
    multi = len(sel) > 1
    for i in self.items:
        i.selectionChanged(i in sel, multi)
    if len(sel) == 1:
        self.multiSelectBox.hide()
        self.ui.mirrorSelectionBtn.hide()
        self.ui.reflectSelectionBtn.hide()
        self.ui.resetTransformsBtn.hide()
    elif len(sel) > 1:
        self.showMultiSelectBox()
    self.sigSelectionChanged.emit(self, sel)