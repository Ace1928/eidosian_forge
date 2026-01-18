import pytest
import numpy as np
from numpy.testing import (
@pytest.mark.parametrize('bytes_', [b'spam', np.array(567.0)])
def test_void_from_byteslike(bytes_):
    res = np.void(bytes_)
    expected = bytes(bytes_)
    assert type(res) is np.void
    assert res.item() == expected
    res = np.void(bytes_, dtype='V100')
    assert type(res) is np.void
    assert res.item()[:len(expected)] == expected
    assert res.item()[len(expected):] == b'\x00' * (res.nbytes - len(expected))
    res = np.void(bytes_, dtype='V4')
    assert type(res) is np.void
    assert res.item() == expected[:4]