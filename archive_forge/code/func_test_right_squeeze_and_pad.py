from collections import OrderedDict
import numpy as np
import pandas as pd
import pytest
from statsmodels.tools.validation import (
from statsmodels.tools.validation.validation import _right_squeeze
def test_right_squeeze_and_pad(self):
    data = np.empty((2, 1, 2))
    a = array_like(data, 'a', ndim=3)
    assert a.shape == (2, 1, 2)
    data = np.empty(2)
    a = array_like(data, 'a', ndim=3)
    assert a.shape == (2, 1, 1)
    data = np.empty((2, 1))
    a = array_like(data, 'a', ndim=3)
    assert a.shape == (2, 1, 1)
    data = np.empty((2, 1, 1, 1))
    a = array_like(data, 'a', ndim=3)
    assert a.shape == (2, 1, 1)
    data = np.empty((2, 1, 1, 2, 1, 1))
    with pytest.raises(ValueError):
        array_like(data, 'a', ndim=3)