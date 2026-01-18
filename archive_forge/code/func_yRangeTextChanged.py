from ...Qt import QtCore, QtGui, QtWidgets
from ...WidgetGroup import WidgetGroup
from . import axisCtrlTemplate_generic as ui_template
import weakref
from .ViewBox import ViewBox
def yRangeTextChanged(self):
    self.ctrl[1].manualRadio.setChecked(True)
    self.view().setYRange(*self._validateRangeText(1), padding=0)