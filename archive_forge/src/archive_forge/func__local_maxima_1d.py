import math
import cupy
from cupy._core._scalar import get_typename
from cupy_backends.cuda.api import runtime
from cupyx import jit
def _local_maxima_1d(x):
    samples = x.shape[0] - 2
    block_sz = 128
    n_blocks = (samples + block_sz - 1) // block_sz
    midpoints = cupy.empty(samples, dtype=cupy.int64)
    left_edges = cupy.empty(samples, dtype=cupy.int64)
    right_edges = cupy.empty(samples, dtype=cupy.int64)
    local_max_kernel = _get_module_func(PEAKS_MODULE, 'local_maxima_1d', x)
    local_max_kernel((n_blocks,), (block_sz,), (x.shape[0], x, midpoints, left_edges, right_edges))
    pos_idx = midpoints > 0
    midpoints = midpoints[pos_idx]
    left_edges = left_edges[pos_idx]
    right_edges = right_edges[pos_idx]
    return (midpoints, left_edges, right_edges)