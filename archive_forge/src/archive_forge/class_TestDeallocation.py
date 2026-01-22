from contextlib import contextmanager
import numpy as np
from numba import cuda
from numba.cuda.testing import (unittest, skip_on_cudasim,
from numba.tests.support import captured_stderr
from numba.core import config
@skip_on_cudasim('not supported on CUDASIM')
@skip_if_external_memmgr('Deallocation specific to Numba memory management')
class TestDeallocation(CUDATestCase):

    def test_max_pending_count(self):
        deallocs = cuda.current_context().memory_manager.deallocations
        deallocs.clear()
        self.assertEqual(len(deallocs), 0)
        for i in range(config.CUDA_DEALLOCS_COUNT):
            cuda.to_device(np.arange(1))
            self.assertEqual(len(deallocs), i + 1)
        cuda.to_device(np.arange(1))
        self.assertEqual(len(deallocs), 0)

    def test_max_pending_bytes(self):
        ctx = cuda.current_context()
        deallocs = ctx.memory_manager.deallocations
        deallocs.clear()
        self.assertEqual(len(deallocs), 0)
        mi = ctx.get_memory_info()
        max_pending = 10 ** 6
        old_ratio = config.CUDA_DEALLOCS_RATIO
        try:
            config.CUDA_DEALLOCS_RATIO = max_pending / mi.total
            self.assertAlmostEqual(deallocs._max_pending_bytes, max_pending, delta=1)
            cuda.to_device(np.ones(max_pending // 2, dtype=np.int8))
            self.assertEqual(len(deallocs), 1)
            cuda.to_device(np.ones(deallocs._max_pending_bytes - deallocs._size, dtype=np.int8))
            self.assertEqual(len(deallocs), 2)
            cuda.to_device(np.ones(1, dtype=np.int8))
            self.assertEqual(len(deallocs), 0)
        finally:
            config.CUDA_DEALLOCS_RATIO = old_ratio