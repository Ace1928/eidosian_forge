import warnings
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from ..affines import from_matvec, to_matvec
from ..orientations import (
from ..testing import deprecated_to, expires
def test_ornt_transform():
    assert_array_equal(ornt_transform([[0, 1], [1, 1], [2, -1]], [[1, 1], [0, 1], [2, 1]]), [[1, 1], [0, 1], [2, -1]])
    assert_array_equal(ornt_transform([[0, 1], [1, 1], [2, 1]], [[2, 1], [0, -1], [1, 1]]), [[1, -1], [2, 1], [0, 1]])
    with pytest.raises(ValueError):
        ornt_transform([[0, 1], [1, 1]], [[0, 1], [1, 1], [2, 1]])
    with pytest.raises(ValueError):
        ornt_transform([[0, 1, 1], [1, 1, 1]], [[0, 1, 1], [1, 1, 1]])
    with pytest.raises(ValueError):
        ornt_transform([[0, 1], [1, 1], [1, 1]], [[0, 1], [1, 1], [2, 1]])