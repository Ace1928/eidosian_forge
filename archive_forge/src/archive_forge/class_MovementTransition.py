from PySide2.QtWidgets import *
from PySide2.QtGui import *
from PySide2.QtCore import *
class MovementTransition(QEventTransition):

    def __init__(self, window):
        super(MovementTransition, self).__init__(window, QEvent.KeyPress)
        self.window = window

    def eventTest(self, event):
        if event.type() == QEvent.StateMachineWrapped and event.event().type() == QEvent.KeyPress:
            key = event.event().key()
            return key == Qt.Key_2 or key == Qt.Key_8 or key == Qt.Key_6 or (key == Qt.Key_4)
        return False

    def onTransition(self, event):
        key = event.event().key()
        if key == Qt.Key_4:
            self.window.movePlayer(self.window.Left)
        if key == Qt.Key_8:
            self.window.movePlayer(self.window.Up)
        if key == Qt.Key_6:
            self.window.movePlayer(self.window.Right)
        if key == Qt.Key_2:
            self.window.movePlayer(self.window.Down)