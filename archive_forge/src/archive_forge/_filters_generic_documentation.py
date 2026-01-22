import cupy
from cupy_backends.cuda.api import runtime
from cupy import _util
from cupyx.scipy.ndimage import _filters_core

    The generic 1d filter is different than all other filters and thus is the
    only filter that doesn't use _generate_nd_kernel() and has a completely
    custom raw kernel.
    