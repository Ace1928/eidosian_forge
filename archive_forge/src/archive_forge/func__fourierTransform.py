import math
import warnings
import bisect
import numpy as np
from .. import debug as debug
from .. import functions as fn
from .. import getConfigOption
from ..Qt import QtCore
from .GraphicsObject import GraphicsObject
from .PlotCurveItem import PlotCurveItem
from .ScatterPlotItem import ScatterPlotItem
def _fourierTransform(self, x, y):
    dx = np.diff(x)
    uniform = not np.any(np.abs(dx - dx[0]) > abs(dx[0]) / 1000.0)
    if not uniform:
        x2 = np.linspace(x[0], x[-1], len(x))
        y = np.interp(x2, x, y)
        x = x2
    n = y.size
    f = np.fft.rfft(y) / n
    d = float(x[-1] - x[0]) / (len(x) - 1)
    x = np.fft.rfftfreq(n, d)
    y = np.abs(f)
    return (x, y)