import pytest
import numpy as np
from numpy.testing import assert_array_equal
from numpy.core._multiarray_umath import (
def test_possible_and_impossible_reduce(self):
    a = self._get_array(2.0)
    res = np.add.reduce(a, initial=0.0)
    assert res == a.astype(np.float64).sum()
    with pytest.raises(TypeError, match='the resolved dtypes are not compatible'):
        np.multiply.reduce(a)