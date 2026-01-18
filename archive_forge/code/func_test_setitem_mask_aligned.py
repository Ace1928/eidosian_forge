import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
@pytest.mark.parametrize('as_callable', [True, False])
@pytest.mark.parametrize('setter', ['loc', None])
def test_setitem_mask_aligned(self, data, as_callable, setter):
    ser = pd.Series(data)
    mask = np.zeros(len(data), dtype=bool)
    mask[:2] = True
    if as_callable:
        mask2 = lambda x: mask
    else:
        mask2 = mask
    if setter:
        target = getattr(ser, setter)
    else:
        target = ser
    target[mask2] = data[5:7]
    ser[mask2] = data[5:7]
    assert ser[0] == data[5]
    assert ser[1] == data[6]