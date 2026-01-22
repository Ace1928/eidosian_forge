import cupy
from cupyx.scipy._lib._util import _asarray_validated, float_factorial
def _isscalar(x):
    """Check whether x is if a scalar type, or 0-dim"""
    return cupy.isscalar(x) or (hasattr(x, 'shape') and x.shape == ())