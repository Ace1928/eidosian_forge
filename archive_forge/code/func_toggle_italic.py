import sys
from PySide2.QtCore import Qt, QRectF
from PySide2.QtGui import QBrush, QColor, QPainter, QPen
from PySide2.QtWidgets import (QApplication, QDoubleSpinBox,
from PySide2.QtCharts import QtCharts
def toggle_italic(self):
    legend = self.chart.legend()
    font = legend.font()
    font.setItalic(not font.italic())
    legend.setFont(font)