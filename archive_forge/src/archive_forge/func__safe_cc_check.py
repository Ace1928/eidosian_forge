import numpy as np
from numba import cuda, int32, float32
from numba.cuda.testing import skip_on_cudasim, unittest, CUDATestCase
from numba.core.config import ENABLE_CUDASIM
def _safe_cc_check(cc):
    if ENABLE_CUDASIM:
        return True
    else:
        return cuda.get_current_device().compute_capability >= cc