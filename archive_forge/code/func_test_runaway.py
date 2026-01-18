from numba import cuda
from numba.core.errors import TypingError
from numba.cuda.testing import CUDATestCase, skip_on_cudasim
import numpy as np
import unittest
@skip_on_cudasim('Simulator does not compile')
def test_runaway(self):
    with self.assertRaises(TypingError) as raises:
        cfunc = self.mod.runaway_self

        @cuda.jit('void()')
        def kernel():
            cfunc(1)
    self.assertIn('cannot type infer runaway recursion', str(raises.exception))