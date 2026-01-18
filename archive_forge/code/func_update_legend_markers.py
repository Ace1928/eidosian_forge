import sys
from PySide2.QtCore import Qt
from PySide2.QtGui import QColor, QFont, QPainter
from PySide2.QtWidgets import QApplication, QMainWindow
from PySide2.QtCharts import QtCharts
def update_legend_markers(self):
    for series in self.series():
        markers = self.legend().markers(series)
        for marker in markers:
            if series == self.main_series:
                marker.setVisible(False)
            else:
                marker.setLabel('{} {:.2f}%'.format(marker.slice().label(), marker.slice().percentage() * 100, 0))
                marker.setFont(QFont('Arial', 8))