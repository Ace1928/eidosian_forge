import json, os, warnings
from PySide2 import QtCore
from PySide2.QtCore import (QDir, QFileInfo, QModelIndex, QStandardPaths, Qt,
from PySide2.QtGui import QIcon, QPixmap, QStandardItem, QStandardItemModel
from PySide2.QtWidgets import (QAction, QDockWidget, QMenu, QMessageBox,
def _populate_actions(self, parent_item, target_object, first_action):
    existing_actions = target_object.actions()
    existing_action_count = len(existing_actions)
    a = first_action
    row_count = parent_item.rowCount()
    for r in range(0, row_count):
        item = parent_item.child(r)
        title = item.text()
        icon = item.icon()
        url = item.data(_url_role)
        if a < existing_action_count:
            action = existing_actions[a]
            if title != action.toolTip():
                action.setText(BookmarkWidget.short_title(title))
                action.setIcon(icon)
                action.setToolTip(title)
                action.setData(url)
                action.setVisible(True)
        else:
            action = target_object.addAction(icon, BookmarkWidget.short_title(title))
            action.setToolTip(title)
            action.setData(url)
            action.triggered.connect(self._action_activated)
        a = a + 1
    while a < existing_action_count:
        existing_actions[a].setVisible(False)
        a = a + 1