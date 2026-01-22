import itertools
import cupy as cp
from cupyx.scipy.interpolate._bspline2 import make_interp_spline
from cupyx.scipy.interpolate._cubic import PchipInterpolator
@staticmethod
def _do_spline_fit(x, y, pt, k):
    local_interp = make_interp_spline(x, y, k=k, axis=0)
    values = local_interp(pt)
    return values