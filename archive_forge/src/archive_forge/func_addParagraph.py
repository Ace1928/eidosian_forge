from PySide2.QtCore import QDate, QFile, Qt, QTextStream
from PySide2.QtGui import (QFont, QIcon, QKeySequence, QTextCharFormat,
from PySide2.QtPrintSupport import QPrintDialog, QPrinter
from PySide2.QtWidgets import (QAction, QApplication, QDialog, QDockWidget,
import dockwidgets_rc
def addParagraph(self, paragraph):
    if not paragraph:
        return
    document = self.textEdit.document()
    cursor = document.find('Yours sincerely,')
    if cursor.isNull():
        return
    cursor.beginEditBlock()
    cursor.movePosition(QTextCursor.PreviousBlock, QTextCursor.MoveAnchor, 2)
    cursor.insertBlock()
    cursor.insertText(paragraph)
    cursor.insertBlock()
    cursor.endEditBlock()