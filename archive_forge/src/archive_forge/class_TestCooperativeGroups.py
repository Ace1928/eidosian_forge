import unittest
from numba.cuda.testing import (CUDATestCase, skip_on_cudasim,
@skip_if_cudadevrt_missing
@skip_unless_cc_60
@skip_if_mvc_enabled('CG not supported with MVC')
@skip_on_cudasim("cudasim doesn't support cuda import at non-top-level")
class TestCooperativeGroups(CUDATestCase):

    def test_ex_grid_sync(self):
        from numba import cuda, int32
        import numpy as np
        sig = (int32[:, ::1],)

        @cuda.jit(sig)
        def sequential_rows(M):
            col = cuda.grid(1)
            g = cuda.cg.this_grid()
            rows = M.shape[0]
            cols = M.shape[1]
            for row in range(1, rows):
                opposite = cols - col - 1
                M[row, col] = M[row - 1, opposite] + 1
                g.sync()
        A = np.zeros((1024, 1024), dtype=np.int32)
        blockdim = 32
        griddim = A.shape[1] // blockdim
        mb = sequential_rows.overloads[sig].max_cooperative_grid_blocks(blockdim)
        if mb < griddim:
            self.skipTest('Device does not support a large enough coop grid')
        sequential_rows[griddim, blockdim](A)
        reference = np.tile(np.arange(1024), (1024, 1)).T
        np.testing.assert_equal(A, reference)