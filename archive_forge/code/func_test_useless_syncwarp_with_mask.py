import numpy as np
from numba import cuda, int32, float32
from numba.cuda.testing import skip_on_cudasim, unittest, CUDATestCase
from numba.core.config import ENABLE_CUDASIM
@skip_on_cudasim('syncwarp not implemented on cudasim')
@unittest.skipUnless(_safe_cc_check((7, 0)), 'Partial masks require CC 7.0 or greater')
def test_useless_syncwarp_with_mask(self):
    self._test_useless(useless_syncwarp_with_mask)