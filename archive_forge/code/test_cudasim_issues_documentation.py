import threading
import numpy as np
from numba import cuda
from numba.cuda.testing import CUDATestCase, skip_unless_cudasim
import numba.cuda.simulator as simulator
import unittest

        Discovered in https://github.com/numba/numba/issues/1837.
        When the `cuda` module is referenced in a device function,
        it does not have the kernel API (e.g. cuda.threadIdx, cuda.shared)
        