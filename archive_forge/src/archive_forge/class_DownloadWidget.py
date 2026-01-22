import sys
from PySide2 import QtCore
from PySide2.QtCore import QDir, QFileInfo, QStandardPaths, Qt, QUrl
from PySide2.QtGui import QDesktopServices
from PySide2.QtWidgets import (QAction, QLabel, QMenu, QProgressBar,
from PySide2.QtWebEngineWidgets import QWebEngineDownloadItem
class DownloadWidget(QProgressBar):
    """Lets you track progress of a QWebEngineDownloadItem."""
    finished = QtCore.Signal()
    remove_requested = QtCore.Signal()

    def __init__(self, download_item):
        super(DownloadWidget, self).__init__()
        self._download_item = download_item
        download_item.finished.connect(self._finished)
        download_item.downloadProgress.connect(self._download_progress)
        download_item.stateChanged.connect(self._update_tool_tip())
        path = download_item.path()
        self.setMaximumWidth(300)
        description = QFileInfo(path).fileName()
        description_length = len(description)
        if description_length > 30:
            description = '{}...{}'.format(description[0:10], description[description_length - 10:])
        self.setFormat('{} %p%'.format(description))
        self.setOrientation(Qt.Horizontal)
        self.setMinimum(0)
        self.setValue(0)
        self.setMaximum(100)
        self._update_tool_tip()
        if sys.platform == 'darwin':
            self.setStyle(QStyleFactory.create('fusion'))

    @staticmethod
    def open_file(file):
        QDesktopServices.openUrl(QUrl.fromLocalFile(file))

    @staticmethod
    def open_download_directory():
        path = QStandardPaths.writableLocation(QStandardPaths.DownloadLocation)
        DownloadWidget.open_file(path)

    def state(self):
        return self._download_item.state()

    def _update_tool_tip(self):
        path = self._download_item.path()
        tool_tip = '{}\n{}'.format(self._download_item.url().toString(), QDir.toNativeSeparators(path))
        total_bytes = self._download_item.total_bytes()
        if total_bytes > 0:
            tool_tip += '\n{}K'.format(total_bytes / 1024)
        state = self.state()
        if state == QWebEngineDownloadItem.DownloadRequested:
            tool_tip += '\n(requested)'
        elif state == QWebEngineDownloadItem.DownloadInProgress:
            tool_tip += '\n(downloading)'
        elif state == QWebEngineDownloadItem.DownloadCompleted:
            tool_tip += '\n(completed)'
        elif state == QWebEngineDownloadItem.DownloadCancelled:
            tool_tip += '\n(cancelled)'
        else:
            tool_tip += '\n(interrupted)'
        self.setToolTip(tool_tip)

    def _download_progress(self, bytes_received, bytes_total):
        self.setValue(int(100 * bytes_received / bytes_total))

    def _finished(self):
        self._update_tool_tip()
        self.finished.emit()

    def _launch(self):
        DownloadWidget.open_file(self._download_item.path())

    def mouse_double_click_event(self, event):
        if self.state() == QWebEngineDownloadItem.DownloadCompleted:
            self._launch()

    def context_menu_event(self, event):
        state = self.state()
        context_menu = QMenu()
        launch_action = context_menu.addAction('Launch')
        launch_action.setEnabled(state == QWebEngineDownloadItem.DownloadCompleted)
        show_in_folder_action = context_menu.addAction('Show in Folder')
        show_in_folder_action.setEnabled(state == QWebEngineDownloadItem.DownloadCompleted)
        cancel_action = context_menu.addAction('Cancel')
        cancel_action.setEnabled(state == QWebEngineDownloadItem.DownloadInProgress)
        remove_action = context_menu.addAction('Remove')
        remove_action.setEnabled(state != QWebEngineDownloadItem.DownloadInProgress)
        chosen_action = context_menu.exec_(event.globalPos())
        if chosen_action == launch_action:
            self._launch()
        elif chosen_action == show_in_folder_action:
            DownloadWidget.open_file(QFileInfo(self._download_item.path()).absolutePath())
        elif chosen_action == cancel_action:
            self._download_item.cancel()
        elif chosen_action == remove_action:
            self.remove_requested.emit()