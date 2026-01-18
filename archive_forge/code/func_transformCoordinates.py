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
def transformCoordinates(tr, coords, transpose=False):
    """
    Map a set of 2D or 3D coordinates through a QTransform or QMatrix4x4.
    The shape of coords must be (2,...) or (3,...)
    The mapping will _ignore_ any perspective transformations.
    
    For coordinate arrays with ndim=2, this is basically equivalent to matrix multiplication.
    Most arrays, however, prefer to put the coordinate axis at the end (eg. shape=(...,3)). To 
    allow this, use transpose=True.
    
    """
    if transpose:
        coords = coords.transpose((coords.ndim - 1,) + tuple(range(0, coords.ndim - 1)))
    nd = coords.shape[0]
    if isinstance(tr, np.ndarray):
        m = tr
    else:
        m = transformToArray(tr)
        m = m[:m.shape[0] - 1]
    if m.shape == (2, 3) and nd == 3:
        m2 = np.zeros((3, 4))
        m2[:2, :2] = m[:2, :2]
        m2[:2, 3] = m[:2, 2]
        m2[2, 2] = 1
        m = m2
    if m.shape == (3, 4) and nd == 2:
        m2 = np.empty((2, 3))
        m2[:, :2] = m[:2, :2]
        m2[:, 2] = m[:2, 3]
        m = m2
    m = m.reshape(m.shape + (1,) * (coords.ndim - 1))
    coords = coords[np.newaxis, ...]
    translate = m[:, -1]
    m = m[:, :-1]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        mapped = (m * coords).sum(axis=1)
    mapped += translate
    if transpose:
        mapped = mapped.transpose(tuple(range(1, mapped.ndim)) + (0,))
    return mapped