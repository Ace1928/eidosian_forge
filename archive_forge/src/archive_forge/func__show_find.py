import sys
from bookmarkwidget import BookmarkWidget
from browsertabwidget import BrowserTabWidget
from downloadwidget import DownloadWidget
from findtoolbar import FindToolBar
from webengineview import QWebEnginePage, WebEngineView
from PySide2 import QtCore
from PySide2.QtCore import Qt, QUrl
from PySide2.QtGui import QCloseEvent, QKeySequence, QIcon
from PySide2.QtWidgets import (qApp, QAction, QApplication, QDesktopWidget,
from PySide2.QtWebEngineWidgets import (QWebEngineDownloadItem, QWebEnginePage,
def _show_find(self):
    if self._find_tool_bar is None:
        self._find_tool_bar = FindToolBar()
        self._find_tool_bar.find.connect(self._tab_widget.find)
        self.addToolBar(Qt.BottomToolBarArea, self._find_tool_bar)
    else:
        self._find_tool_bar.show()
    self._find_tool_bar.focus_find()