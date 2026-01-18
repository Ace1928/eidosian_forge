from ...Qt import QtCore, QtGui, QtWidgets
from ...WidgetGroup import WidgetGroup
from . import axisCtrlTemplate_generic as ui_template
import weakref
from .ViewBox import ViewBox
def xAutoClicked(self):
    val = self.ctrl[0].autoPercentSpin.value() * 0.01
    self.view().enableAutoRange(ViewBox.XAxis, val)