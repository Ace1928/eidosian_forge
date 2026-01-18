import numpy as np
import pytest
from pandas.core.indexers import (
def test_validate_indices_high(self):
    indices = np.asarray([0, 1, 2])
    with pytest.raises(IndexError, match='indices are out'):
        validate_indices(indices, 2)