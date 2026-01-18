import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('index', [Index(list('abcde')), Index(list('abcde'), dtype='category'), date_range('2020-01-01', periods=5), timedelta_range('1 day', periods=5), period_range('2020-01-01', periods=5)])
def test_scalar_non_numeric(self, index, frame_or_series, indexer_sl):
    s = gen_obj(frame_or_series, index)
    with pytest.raises(KeyError, match='^3.0$'):
        indexer_sl(s)[3.0]
    assert 3.0 not in s
    s2 = s.copy()
    indexer_sl(s2)[3.0] = 10
    if indexer_sl is tm.setitem:
        assert 3.0 in s2.axes[-1]
    elif indexer_sl is tm.loc:
        assert 3.0 in s2.axes[0]
    else:
        assert 3.0 not in s2.axes[0]
        assert 3.0 not in s2.axes[-1]