import traceback
import threading
import multiprocessing
import numpy as np
from numba import cuda
from numba.cuda.testing import (skip_on_cudasim, skip_under_cuda_memcheck,
import unittest
@unittest.skipIf(not has_mp_get_context, 'no multiprocessing.get_context')
def test_spawn_concurrent_compilation(self):
    cuda.get_current_device()
    ctx = multiprocessing.get_context('spawn')
    q = ctx.Queue()
    p = ctx.Process(target=spawn_process_entry, args=(q,))
    p.start()
    try:
        err = q.get()
    finally:
        p.join()
    if err is not None:
        raise AssertionError(err)
    self.assertEqual(p.exitcode, 0, 'test failed in child process')