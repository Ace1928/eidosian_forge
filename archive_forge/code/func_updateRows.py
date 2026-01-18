from ..Qt import QtCore, QtWidgets
from . import VerticalLabel
def updateRows(self, rows):
    for r in self.rowNames[:]:
        if r not in rows:
            self.removeRow(r)
    for r in rows:
        if r not in self.rowNames:
            self.addRow(r)