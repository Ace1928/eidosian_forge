import numpy as np
import pytest
from ..volumeutils import DtypeMapper, Recoder, native_code, swapped_code
def test_recoder_4():
    codes = ((1, 'one'), (2, 'two'))
    rc = Recoder(codes, ['code1', 'label'])
    with pytest.raises(AttributeError):
        rc.code
    assert rc.code1[1] == 1
    assert rc.code1['one'] == 1
    assert rc.label[1] == 'one'
    assert rc.label['one'] == 'one'