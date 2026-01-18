import array
import pytest
import rpy2.rinterface as ri
def test_setitem_int_invalid():
    vec = ri.ByteSexpVector((b'a', b'b', b'c'))
    with pytest.raises(ValueError):
        vec[1] = 333