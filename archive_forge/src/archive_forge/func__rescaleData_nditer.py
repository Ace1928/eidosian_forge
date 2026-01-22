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
def _rescaleData_nditer(data_in, scale, offset, work_dtype, out_dtype, clip):
    """Refer to documentation for rescaleData()"""
    data_out = np.empty_like(data_in, dtype=out_dtype)
    fits_int32 = False
    if data_in.dtype.kind in 'ui' and out_dtype.kind in 'ui':
        lim_in = np.iinfo(data_in.dtype)
        lo = offset.item(0) if isinstance(offset, np.number) else offset
        dst_bounds = (scale * (lim_in.min - lo), scale * (lim_in.max - lo))
        if dst_bounds[1] < dst_bounds[0]:
            dst_bounds = (dst_bounds[1], dst_bounds[0])
        lim32 = np.iinfo(np.int32)
        fits_int32 = lim32.min < dst_bounds[0] and dst_bounds[1] < lim32.max
        if fits_int32 and clip is not None:
            clip = [clip_scalar(v, lim32.min, lim32.max) for v in clip]
    it = np.nditer([data_in, data_out], flags=['external_loop', 'buffered'], op_flags=[['readonly'], ['writeonly', 'no_broadcast']], op_dtypes=[None, work_dtype], casting='unsafe', buffersize=32768)
    with it:
        for x, y in it:
            y[...] = x
            y -= offset
            y *= scale
            if clip is not None:
                if fits_int32:
                    yin = y.astype(np.int32)
                else:
                    yin = y
                clip_array(yin, clip[0], clip[1], out=y)
    return data_out