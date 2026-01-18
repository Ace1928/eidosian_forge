import sys
from PySide2.QtCore import Qt, QRectF
from PySide2.QtGui import QBrush, QColor, QPainter, QPen
from PySide2.QtWidgets import (QApplication, QDoubleSpinBox,
from PySide2.QtCharts import QtCharts
def toggle_attached(self):
    legend = self.chart.legend()
    if legend.isAttachedToChart():
        legend.detachFromChart()
        legend.setBackgroundVisible(True)
        legend.setBrush(QBrush(QColor(128, 128, 128, 128)))
        legend.setPen(QPen(QColor(192, 192, 192, 192)))
        self.show_legend_spinbox()
        self.update_legend_layout()
    else:
        legend.attachToChart()
        legend.setBackgroundVisible(False)
        self.hideLegendSpinbox()
    self.update()