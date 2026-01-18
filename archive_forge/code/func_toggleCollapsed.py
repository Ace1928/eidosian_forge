from ..Qt import QtCore, QtGui, QtWidgets
from .PathButton import PathButton
def toggleCollapsed(self):
    self.setCollapsed(not self._collapsed)