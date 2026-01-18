import argparse
import itertools
import sys
import numpy as np
from utils import FrameCounter
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import VideoTemplate_generic as ui_template
def updateScale():
    global ui, levelSpins
    if ui.rgbLevelsCheck.isChecked():
        for s in levelSpins[2:]:
            s.setEnabled(True)
    else:
        for s in levelSpins[2:]:
            s.setEnabled(False)