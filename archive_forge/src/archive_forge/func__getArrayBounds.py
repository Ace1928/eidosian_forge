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
def _getArrayBounds(self, arr, all_finite):
    if not all_finite:
        selection = np.isfinite(arr)
        all_finite = selection.all()
        if not all_finite:
            arr = arr[selection]
    try:
        amin = np.min(arr)
        amax = np.max(arr)
    except ValueError:
        amin = np.nan
        amax = np.nan
    return (amin, amax, all_finite)