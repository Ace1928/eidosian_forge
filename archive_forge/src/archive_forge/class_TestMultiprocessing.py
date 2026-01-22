import os
import multiprocessing as mp
import numpy as np
from numba import cuda
from numba.cuda.testing import skip_on_cudasim, CUDATestCase
import unittest
@skip_on_cudasim('disabled for cudasim')
class TestMultiprocessing(CUDATestCase):

    @unittest.skipUnless(has_mp_get_context, 'requires mp.get_context')
    @unittest.skipUnless(is_unix, 'requires Unix')
    def test_fork(self):
        """
        Test fork detection.
        """
        cuda.current_context()
        ctx = mp.get_context('fork')
        q = ctx.Queue()
        proc = ctx.Process(target=fork_test, args=[q])
        proc.start()
        exc = q.get()
        proc.join()
        self.assertIsNotNone(exc)
        self.assertIn('CUDA initialized before forking', str(exc))