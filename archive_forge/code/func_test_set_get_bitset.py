import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.ensemble._hist_gradient_boosting._bitset import (
from sklearn.ensemble._hist_gradient_boosting.common import X_DTYPE
@pytest.mark.parametrize('values_to_insert, expected_bitset', [([0, 4, 33], np.array([2 ** 0 + 2 ** 4, 2 ** 1, 0], dtype=np.uint32)), ([31, 32, 33, 79], np.array([2 ** 31, 2 ** 0 + 2 ** 1, 2 ** 15], dtype=np.uint32))])
def test_set_get_bitset(values_to_insert, expected_bitset):
    n_32bits_ints = 3
    bitset = np.zeros(n_32bits_ints, dtype=np.uint32)
    for value in values_to_insert:
        set_bitset_memoryview(bitset, value)
    assert_allclose(expected_bitset, bitset)
    for value in range(32 * n_32bits_ints):
        if value in values_to_insert:
            assert in_bitset_memoryview(bitset, value)
        else:
            assert not in_bitset_memoryview(bitset, value)