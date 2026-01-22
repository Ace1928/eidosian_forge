import sys
import re
from PySide2.QtCore import (QFile, Qt, QTextStream)
from PySide2.QtGui import (QColor, QFont, QKeySequence, QSyntaxHighlighter,
from PySide2.QtWidgets import (QAction, qApp, QApplication, QFileDialog, QMainWindow,
import syntaxhighlighter_rc
class Highlighter(QSyntaxHighlighter):

    def __init__(self, parent=None):
        QSyntaxHighlighter.__init__(self, parent)
        self.mappings = {}

    def addMapping(self, pattern, format):
        self.mappings[pattern] = format

    def highlightBlock(self, text):
        for pattern in self.mappings:
            for m in re.finditer(pattern, text):
                s, e = m.span()
                self.setFormat(s, e - s, self.mappings[pattern])