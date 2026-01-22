import sys
from os.path import abspath, dirname, join
from PySide2.QtCore import QObject, Slot
from PySide2.QtGui import QGuiApplication
from PySide2.QtQml import QQmlApplicationEngine
class Bridge(QObject):

    @Slot(str, result=str)
    def getColor(self, s):
        if s.lower() == 'red':
            return '#ef9a9a'
        elif s.lower() == 'green':
            return '#a5d6a7'
        elif s.lower() == 'blue':
            return '#90caf9'
        else:
            return 'white'

    @Slot(float, result=int)
    def getSize(self, s):
        size = int(s * 34)
        if size <= 0:
            return 1
        else:
            return size

    @Slot(str, result=bool)
    def getItalic(self, s):
        if s.lower() == 'italic':
            return True
        else:
            return False

    @Slot(str, result=bool)
    def getBold(self, s):
        if s.lower() == 'bold':
            return True
        else:
            return False

    @Slot(str, result=bool)
    def getUnderline(self, s):
        if s.lower() == 'underline':
            return True
        else:
            return False