import cupy
from cupy import _core
from cupy._core import fusion
from cupy import _util
from cupy._core import _routines_indexing as _indexing
from cupy._core import _routines_statistics as _statistics
`assume_increasing` is used in the kernel to
    skip monotonically increasing or decreasing verification
    inside the cuda kernel.
    