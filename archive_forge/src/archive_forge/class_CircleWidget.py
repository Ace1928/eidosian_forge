from PySide2.QtCore import QRect, QRectF, QSize, Qt, QTimer
from PySide2.QtGui import QColor, QPainter, QPalette, QPen
from PySide2.QtWidgets import (QApplication, QFrame, QGridLayout, QLabel,
class CircleWidget(QWidget):

    def __init__(self, parent=None):
        super(CircleWidget, self).__init__(parent)
        self.floatBased = False
        self.antialiased = False
        self.frameNo = 0
        self.setBackgroundRole(QPalette.Base)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def setFloatBased(self, floatBased):
        self.floatBased = floatBased
        self.update()

    def setAntialiased(self, antialiased):
        self.antialiased = antialiased
        self.update()

    def minimumSizeHint(self):
        return QSize(50, 50)

    def sizeHint(self):
        return QSize(180, 180)

    def nextAnimationFrame(self):
        self.frameNo += 1
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, self.antialiased)
        painter.translate(self.width() / 2, self.height() / 2)
        for diameter in range(0, 256, 9):
            delta = abs(self.frameNo % 128 - diameter / 2)
            alpha = 255 - delta * delta / 4 - diameter
            if alpha > 0:
                painter.setPen(QPen(QColor(0, diameter / 2, 127, alpha), 3))
                if self.floatBased:
                    painter.drawEllipse(QRectF(-diameter / 2.0, -diameter / 2.0, diameter, diameter))
                else:
                    painter.drawEllipse(QRect(-diameter / 2, -diameter / 2, diameter, diameter))