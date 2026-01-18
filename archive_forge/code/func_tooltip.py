import sys
from PySide2.QtWidgets import (QApplication, QWidget, QGraphicsScene,
from PySide2.QtCore import Qt, QPointF, QRectF, QRect
from PySide2.QtCharts import QtCharts
from PySide2.QtGui import QPainter, QFont, QFontMetrics, QPainterPath, QColor
def tooltip(self, point, state):
    if self._tooltip == 0:
        self._tooltip = Callout(self._chart)
    if state:
        self._tooltip.setText('X: {0:.2f} \nY: {1:.2f} '.format(point.x(), point.y()))
        self._tooltip.setAnchor(point)
        self._tooltip.setZValue(11)
        self._tooltip.updateGeometry()
        self._tooltip.show()
    else:
        self._tooltip.hide()