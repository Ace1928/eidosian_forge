import numpy as np
from numpy.testing import (
import pytest
@pytest.mark.parametrize('type_code', TYPE_CODES)
def test_poly1d_misc(self, type_code: str) -> None:
    dtype = np.dtype(type_code)
    ar = np.array([1, 2, 3], dtype=dtype)
    p = np.poly1d(ar)
    assert_equal(np.asarray(p), ar)
    assert_equal(np.asarray(p).dtype, dtype)
    assert_equal(len(p), 2)
    comparison_dct = {-1: 0, 0: 3, 1: 2, 2: 1, 3: 0}
    for index, ref in comparison_dct.items():
        scalar = p[index]
        assert_equal(scalar, ref)
        if dtype == np.object_:
            assert isinstance(scalar, int)
        else:
            assert_equal(scalar.dtype, dtype)