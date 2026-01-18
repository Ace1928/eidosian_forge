import sys
import math, random
from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *
from PySide2.QtOpenGL import *
def updateBrush(self):
    gradient = QRadialGradient(QPointF(self.radius, self.radius), self.radius, QPointF(self.radius * 0.5, self.radius * 0.5))
    gradient.setColorAt(0, QColor(255, 255, 255, 255))
    gradient.setColorAt(0.25, self.innerColor)
    gradient.setColorAt(1, self.outerColor)
    self.brush = QBrush(gradient)