import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
def test_take_empty(self, data, na_value, na_cmp):
    empty = data[:0]
    result = empty.take([-1], allow_fill=True)
    assert na_cmp(result[0], na_value)
    msg = 'cannot do a non-empty take from an empty axes|out of bounds'
    with pytest.raises(IndexError, match=msg):
        empty.take([-1])
    with pytest.raises(IndexError, match='cannot do a non-empty take'):
        empty.take([0, 1])