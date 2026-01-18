import argparse
import itertools
import sys
import numpy as np
from utils import FrameCounter
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import VideoTemplate_generic as ui_template
def updateLUT():
    global LUT, ui
    dtype = ui.dtypeCombo.currentText()
    if dtype == 'uint8':
        n = 256
    else:
        n = 4096
    LUT = ui.gradient.getLookupTable(n, alpha=ui.alphaCheck.isChecked())
    if _has_cupy and xp == cp:
        LUT = cp.asarray(LUT)