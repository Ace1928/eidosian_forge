import numpy as np
import pytest
from pandas.core.indexers import (
class TestValidateIndices:

    def test_validate_indices_ok(self):
        indices = np.asarray([0, 1])
        validate_indices(indices, 2)
        validate_indices(indices[:0], 0)
        validate_indices(np.array([-1, -1]), 0)

    def test_validate_indices_low(self):
        indices = np.asarray([0, -2])
        with pytest.raises(ValueError, match="'indices' contains"):
            validate_indices(indices, 2)

    def test_validate_indices_high(self):
        indices = np.asarray([0, 1, 2])
        with pytest.raises(IndexError, match='indices are out'):
            validate_indices(indices, 2)

    def test_validate_indices_empty(self):
        with pytest.raises(IndexError, match='indices are out'):
            validate_indices(np.array([0, 1]), 0)