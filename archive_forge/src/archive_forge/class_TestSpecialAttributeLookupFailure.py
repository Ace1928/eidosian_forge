from itertools import permutations, product
import pytest
from pytest import param
import numpy as np
from numpy.core._rational_tests import rational
from numpy.core._multiarray_umath import _discover_array_parameters
from numpy.testing import (
class TestSpecialAttributeLookupFailure:

    class WeirdArrayLike:

        @property
        def __array__(self):
            raise RuntimeError('oops!')

    class WeirdArrayInterface:

        @property
        def __array_interface__(self):
            raise RuntimeError('oops!')

    def test_deprecated(self):
        with pytest.raises(RuntimeError):
            np.array(self.WeirdArrayLike())
        with pytest.raises(RuntimeError):
            np.array(self.WeirdArrayInterface())