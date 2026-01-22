from ..Qt import QtCore, QtGui, QtWidgets
import math
import sys
import warnings
import numpy as np
from .. import Qt, debug
from .. import functions as fn
from .. import getConfigOption
from .GraphicsObject import GraphicsObject
def arrayToLineSegments(x, y, connect, finiteCheck, out=None):
    if out is None:
        out = Qt.internals.PrimitiveArray(QtCore.QLineF, 4)
    if len(x) < 2:
        out.resize(0)
        return out
    connect_array = None
    if isinstance(connect, np.ndarray):
        connect_array, connect = (np.asarray(connect[:-1], dtype=bool), 'array')
    all_finite = True
    if finiteCheck or connect == 'finite':
        mask = np.isfinite(x) & np.isfinite(y)
        all_finite = np.all(mask)
    if connect == 'all':
        if not all_finite:
            x = x[mask]
            y = y[mask]
    elif connect == 'finite':
        if all_finite:
            connect = 'all'
        else:
            connect_array = mask[:-1] & mask[1:]
    elif connect in ['pairs', 'array']:
        if not all_finite:
            backfill_idx = fn._compute_backfill_indices(mask)
            x = x[backfill_idx]
            y = y[backfill_idx]
    if connect == 'all':
        nsegs = len(x) - 1
        out.resize(nsegs)
        if nsegs:
            memory = out.ndarray()
            memory[:, 0] = x[:-1]
            memory[:, 2] = x[1:]
            memory[:, 1] = y[:-1]
            memory[:, 3] = y[1:]
    elif connect == 'pairs':
        nsegs = len(x) // 2
        out.resize(nsegs)
        if nsegs:
            memory = out.ndarray()
            memory = memory.reshape((-1, 2))
            memory[:, 0] = x[:nsegs * 2]
            memory[:, 1] = y[:nsegs * 2]
    elif connect_array is not None:
        nsegs = np.count_nonzero(connect_array)
        out.resize(nsegs)
        if nsegs:
            memory = out.ndarray()
            memory[:, 0] = x[:-1][connect_array]
            memory[:, 2] = x[1:][connect_array]
            memory[:, 1] = y[:-1][connect_array]
            memory[:, 3] = y[1:][connect_array]
    else:
        nsegs = 0
        out.resize(nsegs)
    return out