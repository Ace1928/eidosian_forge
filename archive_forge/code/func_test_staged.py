import multiprocessing as mp
import itertools
import traceback
import pickle
import numpy as np
from numba import cuda
from numba.cuda.cudadrv import driver
from numba.cuda.testing import (skip_on_arm, skip_on_cudasim,
from numba.tests.support import linux_only, windows_only
import unittest
def test_staged(self):
    arr = np.arange(10, dtype=np.intp)
    devarr = cuda.to_device(arr)
    mpctx = mp.get_context('spawn')
    result_queue = mpctx.Queue()
    ctx = cuda.current_context()
    ipch = ctx.get_ipc_handle(devarr.gpu_data)
    buf = pickle.dumps(ipch)
    ipch_recon = pickle.loads(buf)
    self.assertIs(ipch_recon.base, None)
    if driver.USE_NV_BINDING:
        self.assertEqual(ipch_recon.handle.reserved, ipch.handle.reserved)
    else:
        self.assertEqual(tuple(ipch_recon.handle), tuple(ipch.handle))
    self.assertEqual(ipch_recon.size, ipch.size)
    for device_num in range(len(cuda.gpus)):
        args = (ipch, device_num, result_queue)
        proc = mpctx.Process(target=staged_ipc_handle_test, args=args)
        proc.start()
        succ, out = result_queue.get()
        proc.join(3)
        if not succ:
            self.fail(out)
        else:
            np.testing.assert_equal(arr, out)