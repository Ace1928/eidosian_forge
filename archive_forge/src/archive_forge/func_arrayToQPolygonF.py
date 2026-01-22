from __future__ import division
import decimal
import math
import re
import struct
import sys
import warnings
from collections import OrderedDict
import numpy as np
from . import Qt, debug, getConfigOption, reload
from .metaarray import MetaArray
from .Qt import QT_LIB, QtCore, QtGui
from .util.cupy_helper import getCupy
from .util.numba_helper import getNumbaFunctions
def arrayToQPolygonF(x, y):
    """
    Utility function to convert two 1D-NumPy arrays representing curve data
    (X-axis, Y-axis data) into a single open polygon (QtGui.PolygonF) object.
    
    Thanks to PythonQwt for making this code available
    
    License/copyright: MIT License © Pierre Raybaut 2020.

    Parameters
    ----------
    x : np.array
        x-axis coordinates for data to be plotted, must have have ndim of 1
    y : np.array
        y-axis coordinates for data to be plotted, must have ndim of 1 and 
        be the same length as x
    
    Returns
    -------
    QPolygonF
        Open QPolygonF object that represents the path looking to be plotted
    
    Raises
    ------
    ValueError
        When xdata or ydata does not meet the required criteria
    """
    if not x.size == y.size == x.shape[0] == y.shape[0]:
        raise ValueError('Arguments must be 1D and the same size')
    size = x.size
    polyline = create_qpolygonf(size)
    memory = ndarray_from_qpolygonf(polyline)
    memory[:, 0] = x
    memory[:, 1] = y
    return polyline