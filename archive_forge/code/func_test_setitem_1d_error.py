import numpy
import pytest
from pandas.core.dtypes.common import is_list_like
import modin.numpy as np
from .utils import assert_scalar_or_array_equal
def test_setitem_1d_error():
    arr = np.array([1, 2, 3, 4, 5])
    with pytest.raises(ValueError, match='could not broadcast'):
        arr[0:5] = [1, 2]