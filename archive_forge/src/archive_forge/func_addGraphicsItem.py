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
def addGraphicsItem(self, item, **opts):
    """Add a new GraphicsItem to the scene at pos.
        Common options are name, pos, scale, and z
        """
    citem = CanvasItem(item, **opts)
    item._canvasItem = citem
    self.addItem(citem)
    return citem