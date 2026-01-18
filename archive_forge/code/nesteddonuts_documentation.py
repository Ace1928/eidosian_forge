import sys
from PySide2.QtCore import Qt, QTimer
from PySide2.QtGui import QPainter
from PySide2.QtWidgets import QApplication, QGridLayout, QWidget
from PySide2.QtCharts import QtCharts
from random import randrange
from functools import partial
PySide2 port of the Nested Donuts example from Qt v5.x