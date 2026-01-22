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
def _arrayToQPath_finite(x, y, isfinite=None):
    n = x.shape[0]
    if n == 0:
        return QtGui.QPainterPath()
    if isfinite is None:
        isfinite = np.isfinite(x) & np.isfinite(y)
    path = QtGui.QPainterPath()
    if hasattr(path, 'reserve'):
        path.reserve(n)
    sidx = np.nonzero(~isfinite)[0] + 1
    xchunks = np.split(x, sidx)
    ychunks = np.split(y, sidx)
    chunks = list(zip(xchunks, ychunks))
    maxlen = max((len(chunk) for chunk in xchunks))
    subpoly = create_qpolygonf(maxlen)
    subarr = ndarray_from_qpolygonf(subpoly)
    if hasattr(subpoly, 'resize'):
        subpoly_resize = subpoly.resize
    else:
        subpoly_resize = lambda n, v=QtCore.QPointF(): subpoly.fill(v, n)
    for xchunk, ychunk in chunks[:-1]:
        lc = len(xchunk)
        if lc <= 1:
            continue
        subpoly_resize(lc)
        subarr[:lc, 0] = xchunk
        subarr[:lc, 1] = ychunk
        subarr[lc - 1] = subarr[lc - 2]
        path.addPolygon(subpoly)
    for xchunk, ychunk in chunks[-1:]:
        lc = len(xchunk)
        if lc <= 1:
            continue
        subpoly_resize(lc)
        subarr[:lc, 0] = xchunk
        subarr[:lc, 1] = ychunk
        path.addPolygon(subpoly)
    return path