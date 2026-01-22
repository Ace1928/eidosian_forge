import os
import re
from ...Qt import QtCore, QtGui, QtWidgets
from ..Parameter import Parameter
from .str import StrParameterItem
def _newResizeEvent(self, ev):
    ret = type(self.displayLabel).resizeEvent(self.displayLabel, ev)
    self.updateDisplayLabel()
    return ret