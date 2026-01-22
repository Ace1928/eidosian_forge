import warnings
import numpy
import cupy
def _init_weights_dtype(input):
    """Initialize filter weights based on the input array.

    This helper is only used during initialization of some internal filters
    like prewitt and sobel to avoid costly double-precision computation.
    """
    if input.dtype.kind == 'c':
        return cupy.promote_types(input.real.dtype, cupy.complex64)
    return cupy.promote_types(input.real.dtype, cupy.float32)