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
def _updateDataRect(self):
    """ 
        Finds bounds of plotable data and stores them as ``dataset._dataRect``, 
        stores information about the presence of nonfinite data points.
            """
    if self.y is None or self.x is None:
        return None
    xmin, xmax, self.xAllFinite = self._getArrayBounds(self.x, self.xAllFinite)
    ymin, ymax, self.yAllFinite = self._getArrayBounds(self.y, self.yAllFinite)
    self._dataRect = QtCore.QRectF(QtCore.QPointF(xmin, ymin), QtCore.QPointF(xmax, ymax))