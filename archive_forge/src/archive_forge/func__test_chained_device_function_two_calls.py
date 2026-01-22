from numba.tests.support import override_config
from numba.cuda.testing import skip_on_cudasim
from numba import cuda
from numba.core import types
from numba.cuda.testing import CUDATestCase
import itertools
import re
import unittest
def _test_chained_device_function_two_calls(self, kernel_debug, f1_debug, f2_debug):

    @cuda.jit(device=True, debug=f2_debug, opt=False)
    def f2(x):
        return x + 1

    @cuda.jit(device=True, debug=f1_debug, opt=False)
    def f1(x, y):
        return x - f2(y)

    @cuda.jit(debug=kernel_debug, opt=False)
    def kernel(x, y):
        f1(x, y)
        f2(x)
    kernel[1, 1](1, 2)