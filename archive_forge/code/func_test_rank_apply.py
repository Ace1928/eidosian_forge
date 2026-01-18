from datetime import datetime
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_rank_apply():
    lev1 = tm.rands_array(10, 100)
    lev2 = tm.rands_array(10, 130)
    lab1 = np.random.randint(0, 100, size=500)
    lab2 = np.random.randint(0, 130, size=500)
    df = DataFrame({'value': np.random.randn(500), 'key1': lev1.take(lab1), 'key2': lev2.take(lab2)})
    result = df.groupby(['key1', 'key2']).value.rank()
    expected = [piece.value.rank() for key, piece in df.groupby(['key1', 'key2'])]
    expected = concat(expected, axis=0)
    expected = expected.reindex(result.index)
    tm.assert_series_equal(result, expected)
    result = df.groupby(['key1', 'key2']).value.rank(pct=True)
    expected = [piece.value.rank(pct=True) for key, piece in df.groupby(['key1', 'key2'])]
    expected = concat(expected, axis=0)
    expected = expected.reindex(result.index)
    tm.assert_series_equal(result, expected)