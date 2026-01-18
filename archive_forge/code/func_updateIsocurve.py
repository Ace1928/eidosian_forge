import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui
def updateIsocurve():
    global isoLine, iso
    iso.setLevel(isoLine.value())