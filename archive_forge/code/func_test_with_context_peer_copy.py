from numba import cuda
import numpy as np
from numba.cuda.testing import skip_on_cudasim, CUDATestCase
import threading
import unittest
@unittest.skipIf(len(cuda.gpus) < 2, 'need more than 1 gpus')
def test_with_context_peer_copy(self):
    with cuda.gpus[0]:
        ctx = cuda.current_context()
        if not ctx.can_access_peer(1):
            self.skipTest('Peer access between GPUs disabled')
    hostarr = np.arange(10, dtype=np.float32)
    with cuda.gpus[0]:
        arr1 = cuda.to_device(hostarr)
    with cuda.gpus[1]:
        arr2 = cuda.to_device(np.zeros_like(hostarr))
    with cuda.gpus[0]:
        arr2.copy_to_device(arr1)
        np.testing.assert_equal(arr2.copy_to_host(), hostarr)