from __future__ import print_function
import numpy as np
from numba import config, cuda, int32
from numba.cuda.testing import (unittest, CUDATestCase, skip_on_cudasim,
@skip_unless_cc_60
def test_sync_group(self):
    A = np.full(1, fill_value=np.nan)
    sync_group[1, 1](A)
    self.assertFalse(np.isnan(A[0]), 'Value was not set')