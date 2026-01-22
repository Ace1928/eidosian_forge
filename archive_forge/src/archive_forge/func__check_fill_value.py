import itertools
import cupy as cp
from cupyx.scipy.interpolate._bspline2 import make_interp_spline
from cupyx.scipy.interpolate._cubic import PchipInterpolator
def _check_fill_value(self, values, fill_value):
    if fill_value is not None:
        fill_value_dtype = cp.asarray(fill_value).dtype
        if hasattr(values, 'dtype') and (not cp.can_cast(fill_value_dtype, values.dtype, casting='same_kind')):
            raise ValueError("fill_value must be either 'None' or of a type compatible with values")
    return fill_value