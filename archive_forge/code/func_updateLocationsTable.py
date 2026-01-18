import sys
from PySide2 import QtCore, QtGui, QtWidgets
def updateLocationsTable(self):
    self.locationsTable.setUpdatesEnabled(False)
    self.locationsTable.setRowCount(0)
    for i in range(2):
        if i == 0:
            if self.scope() == QtCore.QSettings.SystemScope:
                continue
            actualScope = QtCore.QSettings.UserScope
        else:
            actualScope = QtCore.QSettings.SystemScope
        for j in range(2):
            if j == 0:
                if not self.application():
                    continue
                actualApplication = self.application()
            else:
                actualApplication = ''
            settings = QtCore.QSettings(self.format(), actualScope, self.organization(), actualApplication)
            row = self.locationsTable.rowCount()
            self.locationsTable.setRowCount(row + 1)
            item0 = QtWidgets.QTableWidgetItem()
            item0.setText(settings.fileName())
            item1 = QtWidgets.QTableWidgetItem()
            disable = not (settings.childKeys() or settings.childGroups())
            if row == 0:
                if settings.isWritable():
                    item1.setText('Read-write')
                    disable = False
                else:
                    item1.setText('Read-only')
                self.buttonBox.button(QtWidgets.QDialogButtonBox.Ok).setDisabled(disable)
            else:
                item1.setText('Read-only fallback')
            if disable:
                item0.setFlags(item0.flags() & ~QtCore.Qt.ItemIsEnabled)
                item1.setFlags(item1.flags() & ~QtCore.Qt.ItemIsEnabled)
            self.locationsTable.setItem(row, 0, item0)
            self.locationsTable.setItem(row, 1, item1)
    self.locationsTable.setUpdatesEnabled(True)