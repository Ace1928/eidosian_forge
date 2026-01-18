import math
from PySide2 import QtCore, QtGui, QtWidgets
import diagramscene_rc
def textInserted(self, item):
    self.buttonGroup.button(self.InsertTextButton).setChecked(False)
    self.scene.setMode(self.pointerTypeGroup.checkedId())