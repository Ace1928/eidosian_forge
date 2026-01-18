import argparse
import itertools
import numpy as np
from utils import FrameCounter
import pyqtgraph as pg
import pyqtgraph.functions as fn
import pyqtgraph.parametertree as ptree
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
@interactor.decorate(useOpenGL={'readonly': not args.allow_opengl_toggle}, plotMethod={'limits': ['pyqtgraph', 'drawPolyline'], 'type': 'list'}, curvePen={'type': 'pen'})
def updateOptions(curvePen=pg.mkPen(), plotMethod='pyqtgraph', fillLevel=False, enableExperimental=use_opengl, useOpenGL=use_opengl):
    pg.setConfigOption('enableExperimental', enableExperimental)
    pw.useOpenGL(useOpenGL)
    curve.setPen(curvePen)
    curve.setFillLevel(0.0 if fillLevel else None)
    curve.setMethod(plotMethod)