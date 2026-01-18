from __future__ import print_function
import numpy as np
from numba import config, cuda, int32
from numba.cuda.testing import (unittest, CUDATestCase, skip_on_cudasim,
@skip_unless_cc_60
def test_sync_at_matrix_row(self):
    if config.ENABLE_CUDASIM:
        shape = (32, 32)
    else:
        shape = (1024, 1024)
    A = np.zeros(shape, dtype=np.int32)
    blockdim = 32
    griddim = A.shape[1] // blockdim
    sig = (int32[:, ::1],)
    c_sequential_rows = cuda.jit(sig)(sequential_rows)
    overload = c_sequential_rows.overloads[sig]
    mb = overload.max_cooperative_grid_blocks(blockdim)
    if griddim > mb:
        unittest.skip('GPU cannot support enough cooperative grid blocks')
    c_sequential_rows[griddim, blockdim](A)
    reference = np.tile(np.arange(shape[0]), (shape[1], 1)).T
    np.testing.assert_equal(A, reference)