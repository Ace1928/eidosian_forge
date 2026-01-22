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
def _pseudoScatterExact(data, spacing=None, shuffle=True, bidir=False):
    """Works by stacking points up one at a time, searching for the lowest position available at each point.
    
    This method produces nice, smooth results but can be prohibitively slow for large datasets.
    """
    inds = np.arange(len(data))
    if shuffle:
        np.random.shuffle(inds)
    data = data[inds]
    if spacing is None:
        spacing = 2.0 * np.std(data) / len(data) ** 0.5
    s2 = spacing ** 2
    yvals = np.empty(len(data))
    if len(data) == 0:
        return yvals
    yvals[0] = 0
    for i in range(1, len(data)):
        x = data[i]
        x0 = data[:i]
        y0 = yvals[:i]
        y = 0
        dx = (x0 - x) ** 2
        xmask = dx < s2
        if xmask.sum() > 0:
            if bidir:
                dirs = [-1, 1]
            else:
                dirs = [1]
            yopts = []
            for direction in dirs:
                y = 0
                dx2 = dx[xmask]
                dy = (s2 - dx2) ** 0.5
                limits = np.empty((2, len(dy)))
                limits[0] = y0[xmask] - dy
                limits[1] = y0[xmask] + dy
                while True:
                    if direction > 0:
                        mask = limits[1] >= y
                    else:
                        mask = limits[0] <= y
                    limits2 = limits[:, mask]
                    mask = (limits2[0] < y) & (limits2[1] > y)
                    if mask.sum() == 0:
                        break
                    if direction > 0:
                        y = limits2[:, mask].max()
                    else:
                        y = limits2[:, mask].min()
                yopts.append(y)
            if bidir:
                y = yopts[0] if -yopts[0] < yopts[1] else yopts[1]
            else:
                y = yopts[0]
        yvals[i] = y
    return yvals[np.argsort(inds)]