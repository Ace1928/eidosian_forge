import numbers
from typing import List
import numpy as np
from onnx.reference.op_run import OpRun
from onnx.reference.ops.op_resize import _get_all_coords
def _cpp_std_round(self, x):

    def round_single_value(v):
        if v >= 0.0:
            return np.floor(v + 0.5)
        else:
            return np.ceil(v - 0.5)
    if isinstance(x, numbers.Number):
        return round_single_value(x)
    else:
        assert x.ndim == 1
        x_rounded = np.zeros_like(x)
        for i in range(x.shape[0]):
            x_rounded[i] = round_single_value(x[i])
        x_rounded = x_rounded.astype(np.int32)
        return x_rounded