from .. import exporters as exporters
from .. import functions as fn
from ..graphicsItems.PlotItem import PlotItem
from ..graphicsItems.ViewBox import ViewBox
from ..Qt import QtCore, QtWidgets
from . import exportDialogTemplate_generic as ui_template
class FormatExportListWidgetItem(QtWidgets.QListWidgetItem):

    def __init__(self, expClass, *args, **kwargs):
        QtWidgets.QListWidgetItem.__init__(self, *args, **kwargs)
        self.expClass = expClass