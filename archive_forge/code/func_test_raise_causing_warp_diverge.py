import numpy as np
from numba import cuda
from numba.cuda.testing import unittest, xfail_unless_cudasim, CUDATestCase
from numba.core import config
def test_raise_causing_warp_diverge(self):
    """Test case for issue #2655.
        """
    self.case_raise_causing_warp_diverge(with_debug_mode=False)