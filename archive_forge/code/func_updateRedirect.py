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
def updateRedirect(self, *args):
    cname = str(self.ui.redirectCombo.currentText())
    man = CanvasManager.instance()
    if self.ui.redirectCheck.isChecked() and cname != '':
        redirect = man.getCanvas(cname)
    else:
        redirect = None
    if self.redirect is redirect:
        return
    self.redirect = redirect
    if redirect is None:
        self.reclaimItems()
    else:
        self.redirectItems(redirect)