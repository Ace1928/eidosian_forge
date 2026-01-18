import numpy as np
from numba import cuda, vectorize, guvectorize
from numba.np.numpy_support import from_dtype
from numba.cuda.testing import CUDATestCase, skip_on_cudasim
import unittest
@guvectorize([(datetime_t, datetime_t, timedelta_t[:])], '(),()->()', target='cuda')
def timediff(start, end, out):
    out[0] = end - start