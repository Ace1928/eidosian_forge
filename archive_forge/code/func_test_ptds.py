import multiprocessing as mp
import logging
import traceback
from numba.cuda.testing import unittest, CUDATestCase
from numba.cuda.testing import (skip_on_cudasim, skip_with_cuda_python,
from numba.tests.support import linux_only
@skip_with_cuda_python('Function names unchanged for PTDS with NV Binding')
def test_ptds(self):
    ctx = mp.get_context('spawn')
    result_queue = ctx.Queue()
    proc = ctx.Process(target=child_test_wrapper, args=(result_queue,))
    proc.start()
    proc.join()
    success, output = result_queue.get()
    if not success:
        self.fail(output)
    ptds_functions = ('cuMemcpyHtoD_v2_ptds', 'cuLaunchKernel_ptsz', 'cuMemcpyDtoH_v2_ptds')
    for fn in ptds_functions:
        with self.subTest(fn=fn, expected=True):
            self.assertIn(fn, output)
    legacy_functions = ('cuMemcpyHtoD_v2', 'cuLaunchKernel', 'cuMemcpyDtoH_v2')
    for fn in legacy_functions:
        with self.subTest(fn=fn, expected=False):
            fn_at_end = f'{fn}\n'
            self.assertNotIn(fn_at_end, output)