import numpy as np
from .._shared.utils import _supported_float_type
from .colorconv import lab2lch, _cart2polar_2pi
def _float_inputs(lab1, lab2, allow_float32=True):
    lab1 = np.asarray(lab1)
    lab2 = np.asarray(lab2)
    if allow_float32:
        float_dtype = _supported_float_type((lab1.dtype, lab2.dtype))
    else:
        float_dtype = np.float64
    lab1 = lab1.astype(float_dtype, copy=False)
    lab2 = lab2.astype(float_dtype, copy=False)
    return (lab1, lab2)