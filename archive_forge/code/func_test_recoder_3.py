import numpy as np
import pytest
from ..volumeutils import DtypeMapper, Recoder, native_code, swapped_code
def test_recoder_3():
    codes = ((1, 'one'), (2, 'two'))
    rc = Recoder(codes)
    assert rc.code[1] == 1
    assert rc.code[2] == 2
    with pytest.raises(KeyError):
        rc.code[3]
    assert rc.code['one'] == 1
    assert rc.code['two'] == 2
    with pytest.raises(KeyError):
        rc.code['three']
    with pytest.raises(AttributeError):
        rc.label