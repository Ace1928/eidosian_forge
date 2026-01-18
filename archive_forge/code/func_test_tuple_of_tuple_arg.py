import numpy as np
from collections import namedtuple
from numba import void, int32, float32, float64
from numba import guvectorize
from numba import cuda
from numba.cuda.testing import skip_on_cudasim, CUDATestCase
import unittest
import warnings
from numba.core.errors import NumbaPerformanceWarning
from numba.tests.support import override_config
def test_tuple_of_tuple_arg(self):
    a = ((1.0, 2.0, 3.0), (4.0, 5.0, 6.0))
    b = ((1.5, 2.5, 3.5), (4.5, 5.5, 6.5))
    self.check_tuple_arg(a, b)