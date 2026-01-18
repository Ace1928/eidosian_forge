from ...Qt import QtCore, QtGui, QtWidgets
from ...WidgetGroup import WidgetGroup
from . import axisCtrlTemplate_generic as ui_template
import weakref
from .ViewBox import ViewBox
def xAutoSpinChanged(self, val):
    self.ctrl[0].autoRadio.setChecked(True)
    self.view().enableAutoRange(ViewBox.XAxis, val * 0.01)