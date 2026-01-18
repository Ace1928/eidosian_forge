from ...Qt import QtCore, QtGui, QtWidgets
from ...WidgetGroup import WidgetGroup
from . import axisCtrlTemplate_generic as ui_template
import weakref
from .ViewBox import ViewBox
def yAutoClicked(self):
    val = self.ctrl[1].autoPercentSpin.value() * 0.01
    self.view().enableAutoRange(ViewBox.YAxis, val)