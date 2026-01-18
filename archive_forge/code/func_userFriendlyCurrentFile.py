from PySide2.QtCore import (QFile, QFileInfo, QPoint, QSettings, QSignalMapper,
from PySide2.QtGui import QIcon, QKeySequence
from PySide2.QtWidgets import (QAction, QApplication, QFileDialog, QMainWindow,
import mdi_rc
def userFriendlyCurrentFile(self):
    return self.strippedName(self.curFile)