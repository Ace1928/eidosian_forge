from PySide2.QtWidgets import (QMainWindow, QAction, QFileDialog, QApplication)
from addresswidget import AddressWidget
def updateActions(self, selection):
    """ Only allow the user to remove or edit an item if an item
            is actually selected.
        """
    indexes = selection.indexes()
    if len(indexes) > 0:
        self.removeAction.setEnabled(True)
        self.editAction.setEnabled(True)
    else:
        self.removeAction.setEnabled(False)
        self.editAction.setEnabled(False)