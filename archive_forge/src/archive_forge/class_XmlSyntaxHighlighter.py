from PySide2 import QtCore, QtGui, QtWidgets, QtXmlPatterns
import schema_rc
from ui_schema import Ui_SchemaMainWindow
class XmlSyntaxHighlighter(QtGui.QSyntaxHighlighter):

    def __init__(self, parent=None):
        super(XmlSyntaxHighlighter, self).__init__(parent)
        self.highlightingRules = []
        format = QtGui.QTextCharFormat()
        format.setForeground(QtCore.Qt.darkBlue)
        format.setFontWeight(QtGui.QFont.Bold)
        pattern = QtCore.QRegExp('(<[a-zA-Z:]+\\b|<\\?[a-zA-Z:]+\\b|\\?>|>|/>|</[a-zA-Z:]+>)')
        self.highlightingRules.append((pattern, format))
        format = QtGui.QTextCharFormat()
        format.setForeground(QtCore.Qt.darkGreen)
        pattern = QtCore.QRegExp('[a-zA-Z:]+=')
        self.highlightingRules.append((pattern, format))
        format = QtGui.QTextCharFormat()
        format.setForeground(QtCore.Qt.red)
        pattern = QtCore.QRegExp('("[^"]*"|\'[^\']*\')')
        self.highlightingRules.append((pattern, format))
        self.commentFormat = QtGui.QTextCharFormat()
        self.commentFormat.setForeground(QtCore.Qt.lightGray)
        self.commentFormat.setFontItalic(True)
        self.commentStartExpression = QtCore.QRegExp('<!--')
        self.commentEndExpression = QtCore.QRegExp('-->')

    def highlightBlock(self, text):
        for pattern, format in self.highlightingRules:
            expression = QtCore.QRegExp(pattern)
            index = expression.indexIn(text)
            while index >= 0:
                length = expression.matchedLength()
                self.setFormat(index, length, format)
                index = expression.indexIn(text, index + length)
        self.setCurrentBlockState(0)
        startIndex = 0
        if self.previousBlockState() != 1:
            startIndex = self.commentStartExpression.indexIn(text)
        while startIndex >= 0:
            endIndex = self.commentEndExpression.indexIn(text, startIndex)
            if endIndex == -1:
                self.setCurrentBlockState(1)
                commentLength = text.length() - startIndex
            else:
                commentLength = endIndex - startIndex + self.commentEndExpression.matchedLength()
            self.setFormat(startIndex, commentLength, self.commentFormat)
            startIndex = self.commentStartExpression.indexIn(text, startIndex + commentLength)