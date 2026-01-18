import pytest
import numpy as np
from numpy.testing import (
def test_void_arraylike_trumps_byteslike():
    m = memoryview(b'just one mintleaf?')
    res = np.void(m)
    assert type(res) is np.ndarray
    assert res.dtype == 'V1'
    assert res.shape == (18,)