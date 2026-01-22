import warnings
import numpy
import cupy
from cupy import _core
from cupy import _util
def _kernel_count():
    return _core.ElementwiseKernel('', 'raw Y y, raw int32 count', '\n        if (y[i] < 0) continue;\n        int j = i;\n        while (j != y[j]) { j = y[j]; }\n        if (j != i) y[i] = j;\n        else atomicAdd(&count[0], 1);\n        ', 'cupyx_scipy_ndimage_label_count')