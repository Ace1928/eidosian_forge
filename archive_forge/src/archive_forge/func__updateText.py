import numpy as np
from ..Qt import QtCore, QtGui, QtWidgets
def _updateText(self):
    self._blockValueChange = True
    try:
        self._text = self.format()
        self.setText(self._text)
    finally:
        self._blockValueChange = False