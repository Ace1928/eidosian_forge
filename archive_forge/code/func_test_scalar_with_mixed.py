import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_scalar_with_mixed(self, indexer_sl):
    s2 = Series([1, 2, 3], index=['a', 'b', 'c'])
    s3 = Series([1, 2, 3], index=['a', 'b', 1.5])
    with pytest.raises(KeyError, match='^1.0$'):
        indexer_sl(s2)[1.0]
    with pytest.raises(KeyError, match='^1\\.0$'):
        indexer_sl(s2)[1.0]
    result = indexer_sl(s2)['b']
    expected = 2
    assert result == expected
    with pytest.raises(KeyError, match='^1.0$'):
        indexer_sl(s3)[1.0]
    if indexer_sl is not tm.loc:
        msg = 'Series.__getitem__ treating keys as positions is deprecated'
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = s3[1]
        expected = 2
        assert result == expected
    with pytest.raises(KeyError, match='^1\\.0$'):
        indexer_sl(s3)[1.0]
    result = indexer_sl(s3)[1.5]
    expected = 3
    assert result == expected