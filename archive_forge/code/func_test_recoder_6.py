import numpy as np
import pytest
from ..volumeutils import DtypeMapper, Recoder, native_code, swapped_code
def test_recoder_6():
    codes = ((1, 'one', '1', 'first'), (2, 'two'))
    rc = Recoder(codes, ['code1', 'label'])
    assert rc.code1[1] == 1
    assert rc.code1['first'] == 1
    assert rc.label[1] == 'one'
    assert rc.label['first'] == 'one'
    with pytest.raises(KeyError):
        Recoder(codes, ['field1'])