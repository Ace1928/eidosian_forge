from ...Qt import QtCore, QtGui, QtWidgets
from ...WidgetGroup import WidgetGroup
from . import axisCtrlTemplate_generic as ui_template
import weakref
from .ViewBox import ViewBox
def _validateRangeText(self, axis):
    """Validate range text inputs. Return current value(s) if invalid."""
    inputs = (self.ctrl[axis].minText.text(), self.ctrl[axis].maxText.text())
    vals = self.view().viewRange()[axis]
    for i, text in enumerate(inputs):
        try:
            vals[i] = float(text)
        except ValueError:
            pass
    return vals