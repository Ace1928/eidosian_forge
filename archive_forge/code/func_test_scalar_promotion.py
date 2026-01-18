from itertools import permutations, product
import pytest
from pytest import param
import numpy as np
from numpy.core._rational_tests import rational
from numpy.core._multiarray_umath import _discover_array_parameters
from numpy.testing import (
@pytest.mark.filterwarnings('ignore:Promotion of numbers:FutureWarning')
def test_scalar_promotion(self):
    for sc1, sc2 in product(scalar_instances(), scalar_instances()):
        sc1, sc2 = (sc1.values[0], sc2.values[0])
        try:
            arr = np.array([sc1, sc2])
        except (TypeError, ValueError):
            continue
        assert arr.shape == (2,)
        try:
            dt1, dt2 = (sc1.dtype, sc2.dtype)
            expected_dtype = np.promote_types(dt1, dt2)
            assert arr.dtype == expected_dtype
        except TypeError as e:
            assert arr.dtype == np.dtype('O')