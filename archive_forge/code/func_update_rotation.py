import sys
from PySide2.QtCore import Qt, QTimer
from PySide2.QtGui import QPainter
from PySide2.QtWidgets import QApplication, QGridLayout, QWidget
from PySide2.QtCharts import QtCharts
from random import randrange
from functools import partial
def update_rotation(self):
    for donut in self.donuts:
        phase_shift = randrange(-50, 100)
        donut.setPieStartAngle(donut.pieStartAngle() + phase_shift)
        donut.setPieEndAngle(donut.pieEndAngle() + phase_shift)