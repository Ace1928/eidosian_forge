from PySide2.QtCore import (QFile, QFileInfo, QPoint, QSettings, QSignalMapper,
from PySide2.QtGui import QIcon, QKeySequence
from PySide2.QtWidgets import (QAction, QApplication, QFileDialog, QMainWindow,
import mdi_rc
def updateWindowMenu(self):
    self.windowMenu.clear()
    self.windowMenu.addAction(self.closeAct)
    self.windowMenu.addAction(self.closeAllAct)
    self.windowMenu.addSeparator()
    self.windowMenu.addAction(self.tileAct)
    self.windowMenu.addAction(self.cascadeAct)
    self.windowMenu.addSeparator()
    self.windowMenu.addAction(self.nextAct)
    self.windowMenu.addAction(self.previousAct)
    self.windowMenu.addAction(self.separatorAct)
    windows = self.mdiArea.subWindowList()
    self.separatorAct.setVisible(len(windows) != 0)
    for i, window in enumerate(windows):
        child = window.widget()
        text = '%d %s' % (i + 1, child.userFriendlyCurrentFile())
        if i < 9:
            text = '&' + text
        action = self.windowMenu.addAction(text)
        action.setCheckable(True)
        action.setChecked(child is self.activeMdiChild())
        action.triggered.connect(self.windowMapper.map)
        self.windowMapper.setMapping(action, window)