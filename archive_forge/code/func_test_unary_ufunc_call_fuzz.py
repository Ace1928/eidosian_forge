import itertools
import pytest
import numpy as np
from numpy.core._multiarray_tests import solve_diophantine, internal_overlap
from numpy.core import _umath_tests
from numpy.lib.stride_tricks import as_strided
from numpy.testing import (
@pytest.mark.slow
def test_unary_ufunc_call_fuzz(self):
    self.check_unary_fuzz(np.invert, None, np.int16)