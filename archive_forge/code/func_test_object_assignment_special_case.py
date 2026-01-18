from itertools import permutations, product
import pytest
from pytest import param
import numpy as np
from numpy.core._rational_tests import rational
from numpy.core._multiarray_umath import _discover_array_parameters
from numpy.testing import (
@pytest.mark.parametrize('arraylike', arraylikes())
@pytest.mark.parametrize('arr', [np.array(0.0), np.arange(4)])
def test_object_assignment_special_case(self, arraylike, arr):
    obj = arraylike(arr)
    empty = np.arange(1, dtype=object)
    empty[:] = [obj]
    assert empty[0] is obj