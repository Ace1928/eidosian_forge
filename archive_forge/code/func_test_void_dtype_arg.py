import pytest
import numpy as np
from numpy.testing import (
def test_void_dtype_arg():
    res = np.void((1, 2), dtype='i,i')
    assert res.item() == (1, 2)
    res = np.void((2, 3), 'i,i')
    assert res.item() == (2, 3)