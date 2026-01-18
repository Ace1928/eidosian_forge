from numba.cuda.testing import skip_on_cudasim, unittest, CUDATestCase
import numpy as np
from numba import config, cuda, njit, types

    A half-open interval on the real number line.
    