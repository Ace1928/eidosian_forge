import csv
import gzip
import os
from math import asin, atan2, cos, degrees, hypot, sin, sqrt
import numpy as np
import pyqtgraph as pg
from pyqtgraph import Point
from pyqtgraph.Qt import QtCore, QtGui
def updateSurfaces(self):
    self.surfaces[0].setParams(self['r1'], self['d1'])
    self.surfaces[1].setParams(-self['r2'], self['d2'])
    self.surfaces[0].setPos(self['x1'], 0)
    self.surfaces[1].setPos(self['x2'], 0)
    self.path = QtGui.QPainterPath()
    self.path.connectPath(self.surfaces[0].path.translated(self.surfaces[0].pos()))
    self.path.connectPath(self.surfaces[1].path.translated(self.surfaces[1].pos()).toReversed())
    self.path.closeSubpath()