import math
from functools import partial
import numpy as np
import param
from ..core import NdOverlay, Overlay
from ..element.chart import Area
from .resample import ResampleOperation1D
def _lttb_inner(x, y, n_out, sampled_x, offset):
    a = 0
    for i in range(n_out - 3):
        o0, o1, o2 = (offset[i], offset[i + 1], offset[i + 2])
        a = _argmax_area(x[a], y[a], x[o1:o2].mean(), y[o1:o2].mean(), x[o0:o1], y[o0:o1]) + offset[i]
        sampled_x[i + 1] = a
    sampled_x[-2] = _argmax_area(x[a], y[a], x[-1], y[-1], x[offset[-2]:offset[-1]], y[offset[-2]:offset[-1]]) + offset[-2]