from __future__ import print_function
import os
import sys
from PySide2.QtCore import QObject, QUrl, Slot
from PySide2.QtGui import QGuiApplication
import PySide2.QtQml
from PySide2.QtQuick import QQuickView
class RotateValue(QObject):

    def __init__(self):
        super(RotateValue, self).__init__()
        self.r = 0

    @Slot(result=int)
    def val(self):
        self.r = self.r + 10
        return self.r