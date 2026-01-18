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
def try_fastpath_argb(xp, ain, aout, useRGBA):
    can_handle = xp is np and ain.dtype == xp.ubyte and ain.flags['C_CONTIGUOUS']
    if not can_handle:
        return False
    nrows, ncols = ain.shape[:2]
    nchans = 1 if ain.ndim == 2 else ain.shape[2]
    Format = QtGui.QImage.Format
    if nchans == 1:
        in_fmt = Format.Format_Grayscale8
    elif nchans == 3:
        in_fmt = Format.Format_RGB888
    else:
        in_fmt = Format.Format_RGBA8888
    if useRGBA:
        out_fmt = Format.Format_RGBA8888
    else:
        out_fmt = Format.Format_ARGB32
    if in_fmt == out_fmt:
        aout[:] = ain
        return True
    npixels_chunk = 512 * 1024
    batch = int(npixels_chunk / ncols / nchans)
    batch = max(1, batch)
    row_beg = 0
    while row_beg < nrows:
        row_end = min(row_beg + batch, nrows)
        ain_view = ain[row_beg:row_end, ...]
        aout_view = aout[row_beg:row_end, ...]
        qimg = QtGui.QImage(ain_view, ncols, ain_view.shape[0], ain.strides[0], in_fmt)
        qimg = qimg.convertToFormat(out_fmt)
        aout_view[:] = imageToArray(qimg, copy=False, transpose=False)
        row_beg = row_end
    return True