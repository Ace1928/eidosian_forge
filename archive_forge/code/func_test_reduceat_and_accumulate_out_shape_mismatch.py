import warnings
import itertools
import sys
import ctypes as ct
import pytest
from pytest import param
import numpy as np
import numpy.core._umath_tests as umt
import numpy.linalg._umath_linalg as uml
import numpy.core._operand_flag_tests as opflag_tests
import numpy.core._rational_tests as _rational_tests
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
from numpy.compat import pickle
@pytest.mark.parametrize('with_cast', [True, False])
def test_reduceat_and_accumulate_out_shape_mismatch(self, with_cast):
    arr = np.arange(5)
    out = np.arange(3)
    if with_cast:
        out = out.astype(np.float64)
    with pytest.raises(ValueError, match='(shape|size)'):
        np.add.reduceat(arr, [0, 3], out=out)
    with pytest.raises(ValueError, match='(shape|size)'):
        np.add.accumulate(arr, out=out)