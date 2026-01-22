import json, os, warnings
from PySide2 import QtCore
from PySide2.QtCore import (QDir, QFileInfo, QModelIndex, QStandardPaths, Qt,
from PySide2.QtGui import QIcon, QPixmap, QStandardItem, QStandardItemModel
from PySide2.QtWidgets import (QAction, QDockWidget, QMenu, QMessageBox,
def _create_item(url, title, icon):
    result = QStandardItem(title)
    result.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
    result.setData(url, _url_role)
    if icon is not None:
        result.setIcon(icon)
    return result