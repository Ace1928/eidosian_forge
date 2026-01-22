from PySide2.QtWidgets import *
from PySide2.QtGui import *
from PySide2.QtCore import *
class LightWidget(QWidget):

    def __init__(self, color):
        super(LightWidget, self).__init__()
        self.color = color
        self.onVal = False

    def isOn(self):
        return self.onVal

    def setOn(self, on):
        if self.onVal == on:
            return
        self.onVal = on
        self.update()

    @Slot()
    def turnOff(self):
        self.setOn(False)

    @Slot()
    def turnOn(self):
        self.setOn(True)

    def paintEvent(self, e):
        if not self.onVal:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(self.color)
        painter.drawEllipse(0, 0, self.width(), self.height())
    on = Property(bool, isOn, setOn)