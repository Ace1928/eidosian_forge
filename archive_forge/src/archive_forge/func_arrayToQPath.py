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
def arrayToQPath(x, y, connect='all', finiteCheck=True):
    """
    Convert an array of x,y coordinates to QPainterPath as efficiently as
    possible. The *connect* argument may be 'all', indicating that each point
    should be connected to the next; 'pairs', indicating that each pair of
    points should be connected, or an array of int32 values (0 or 1) indicating
    connections.
    
    Parameters
    ----------
    x : np.ndarray
        x-values to be plotted of shape (N,)
    y : np.ndarray
        y-values to be plotted, must be same length as `x` of shape (N,)
    connect : {'all', 'pairs', 'finite', (N,) ndarray}, optional
        Argument detailing how to connect the points in the path. `all` will 
        have sequential points being connected.  `pairs` generates lines
        between every other point.  `finite` only connects points that are
        finite.  If an ndarray is passed, containing int32 values of 0 or 1,
        only values with 1 will connect to the previous point.  Def
    finiteCheck : bool, default True
        When false, the check for finite values will be skipped, which can
        improve performance. If nonfinite values are present in `x` or `y`,
        an empty QPainterPath will be generated.
    
    Returns
    -------
    QPainterPath
        QPainterPath object to be drawn
    
    Raises
    ------
    ValueError
        Raised when the connect argument has an invalid value placed within.

    Notes
    -----
    A QPainterPath is generated through one of two ways.  When the connect
    parameter is 'all', a QPolygonF object is created, and
    ``QPainterPath.addPolygon()`` is called.  For other connect parameters
    a ``QDataStream`` object is created and the QDataStream >> QPainterPath
    operator is used to pass the data.  The memory format is as follows

    numVerts(i4)
    0(i4)   x(f8)   y(f8)    <-- 0 means this vertex does not connect
    1(i4)   x(f8)   y(f8)    <-- 1 means this vertex connects to the previous vertex
    ...
    cStart(i4)   fillRule(i4)
    
    see: https://github.com/qt/qtbase/blob/dev/src/gui/painting/qpainterpath.cpp

    All values are big endian--pack using struct.pack('>d') or struct.pack('>i')
    This binary format may change in future versions of Qt
    """
    n = x.shape[0]
    if n == 0:
        return QtGui.QPainterPath()
    connect_array = None
    if isinstance(connect, np.ndarray):
        connect_array, connect = (connect, 'array')
    isfinite = None
    if connect == 'finite':
        if not finiteCheck:
            return _arrayToQPath_finite(x, y)
        isfinite = np.isfinite(x) & np.isfinite(y)
        nonfinite_cnt = n - np.sum(isfinite)
        all_isfinite = nonfinite_cnt == 0
        if all_isfinite:
            connect = 'all'
            finiteCheck = False
        elif nonfinite_cnt / n < 2 / 100:
            return _arrayToQPath_finite(x, y, isfinite)
        else:
            connect = 'array'
            connect_array = isfinite
    if connect == 'all':
        return _arrayToQPath_all(x, y, finiteCheck)
    path = QtGui.QPainterPath()
    if hasattr(path, 'reserve'):
        path.reserve(n)
    if hasattr(path, 'reserve') and getConfigOption('enableExperimental'):
        backstore = None
        arr = Qt.internals.get_qpainterpath_element_array(path, n)
    else:
        backstore = QtCore.QByteArray()
        backstore.resize(4 + n * 20 + 8)
        backstore.replace(0, 4, struct.pack('>i', n))
        backstore.replace(4 + n * 20, 8, struct.pack('>ii', 0, 0))
        arr = np.frombuffer(backstore, dtype=[('c', '>i4'), ('x', '>f8'), ('y', '>f8')], count=n, offset=4)
    backfill_idx = None
    if finiteCheck:
        if isfinite is None:
            isfinite = np.isfinite(x) & np.isfinite(y)
            all_isfinite = np.all(isfinite)
        if not all_isfinite:
            backfill_idx = _compute_backfill_indices(isfinite)
    if backfill_idx is None:
        arr['x'] = x
        arr['y'] = y
    else:
        arr['x'] = x[backfill_idx]
        arr['y'] = y[backfill_idx]
    if connect == 'pairs':
        arr['c'][0::2] = 0
        arr['c'][1::2] = 1
    elif connect == 'array':
        arr['c'][:1] = 0
        arr['c'][1:] = connect_array[:-1]
    else:
        raise ValueError('connect argument must be "all", "pairs", "finite", or array')
    if isinstance(backstore, QtCore.QByteArray):
        ds = QtCore.QDataStream(backstore)
        ds >> path
    return path