import math
from PySide2 import QtCore, QtGui, QtWidgets
import diagramscene_rc
def updatePosition(self):
    line = QtCore.QLineF(self.mapFromItem(self.myStartItem, 0, 0), self.mapFromItem(self.myEndItem, 0, 0))
    self.setLine(line)