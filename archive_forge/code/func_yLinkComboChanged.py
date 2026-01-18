from ...Qt import QtCore, QtGui, QtWidgets
from ...WidgetGroup import WidgetGroup
from . import axisCtrlTemplate_generic as ui_template
import weakref
from .ViewBox import ViewBox
def yLinkComboChanged(self, ind):
    self.view().setYLink(str(self.ctrl[1].linkCombo.currentText()))