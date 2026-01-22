from PySide2.QtWebEngineWidgets import (QWebEnginePage, QWebEngineView,
from PySide2.QtWidgets import QApplication, QDesktopWidget, QTreeView
from PySide2.QtCore import (Signal, QAbstractTableModel, QModelIndex, Qt,
def _activated(self, index):
    item = self._model.item_at(index)
    self.open_url.emit(item.url())