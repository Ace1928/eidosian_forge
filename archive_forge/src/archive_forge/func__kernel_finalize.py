import warnings
import numpy
import cupy
from cupy import _core
from cupy import _util
def _kernel_finalize():
    return _core.ElementwiseKernel('int32 maxlabel', 'raw int32 labels, raw Y y', '\n        if (y[i] < 0) {\n            y[i] = 0;\n            continue;\n        }\n        int yi = y[i];\n        int j_min = 0;\n        int j_max = maxlabel - 1;\n        int j = (j_min + j_max) / 2;\n        while (j_min < j_max) {\n            if (yi == labels[j]) break;\n            if (yi < labels[j]) j_max = j - 1;\n            else j_min = j + 1;\n            j = (j_min + j_max) / 2;\n        }\n        y[i] = j + 1;\n        ', 'cupyx_scipy_ndimage_label_finalize')