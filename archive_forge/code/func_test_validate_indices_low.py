import numpy as np
import pytest
from pandas.core.indexers import (
def test_validate_indices_low(self):
    indices = np.asarray([0, -2])
    with pytest.raises(ValueError, match="'indices' contains"):
        validate_indices(indices, 2)