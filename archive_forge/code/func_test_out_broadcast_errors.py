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
@pytest.mark.parametrize(['arr', 'out'], [([2], np.empty(())), ([1, 2], np.empty(1)), (np.ones((4, 3)), np.empty((4, 1)))], ids=['(1,)->()', '(2,)->(1,)', '(4, 3)->(4, 1)'])
def test_out_broadcast_errors(self, arr, out):
    with pytest.raises(ValueError, match='non-broadcastable'):
        np.positive(arr, out=out)
    with pytest.raises(ValueError, match='non-broadcastable'):
        np.add(np.ones(()), arr, out=out)