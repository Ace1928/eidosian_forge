import sys
from PySide2.QtCore import Qt, QRectF
from PySide2.QtGui import QBrush, QColor, QPainter, QPen
from PySide2.QtWidgets import (QApplication, QDoubleSpinBox,
from PySide2.QtCharts import QtCharts
def toggle_bold(self):
    legend = self.chart.legend()
    font = legend.font()
    font.setBold(not font.bold())
    legend.setFont(font)