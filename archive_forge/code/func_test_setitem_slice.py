import array
import pytest
import rpy2.rinterface as ri
def test_setitem_slice():
    values = (b'a', b'b', b'c')
    vec = ri.ByteSexpVector(values)
    vec[:2] = b'yz'
    assert tuple(vec) == tuple(b'yzc')