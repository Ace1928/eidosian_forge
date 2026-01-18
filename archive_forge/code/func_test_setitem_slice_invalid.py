import array
import pytest
import rpy2.rinterface as ri
def test_setitem_slice_invalid():
    values = (b'a', b'b', b'c')
    vec = ri.ByteSexpVector(values)
    with pytest.raises(TypeError):
        vec['foo'] = (333, ord(b'z'))
    with pytest.raises(ValueError):
        vec[:2] = (333, ord(b'z'))