import math
from PySide2 import QtCore, QtGui, QtWidgets
import diagramscene_rc
def textButtonTriggered(self):
    self.scene.setTextColor(QtGui.QColor(self.textAction.data()))