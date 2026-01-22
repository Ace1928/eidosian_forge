from PySide2 import QtCore
from PySide2.QtCore import Qt, QUrl
from PySide2.QtGui import QIcon, QKeySequence
from PySide2.QtWidgets import (QAction, QCheckBox, QDockWidget, QHBoxLayout,
from PySide2.QtWebEngineWidgets import QWebEnginePage
def _emit_find(self, backward):
    needle = self._line_edit.text().strip()
    if needle:
        flags = QWebEnginePage.FindFlags()
        if self._case_sensitive_checkbox.isChecked():
            flags |= QWebEnginePage.FindCaseSensitively
        if backward:
            flags |= QWebEnginePage.FindBackward
        self.find.emit(needle, flags)