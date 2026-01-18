import numpy as np
import pytest
from pandas._libs.tslibs import Timestamp
import pandas as pd
from pandas import (
from pandas.tests.io.pytables.common import (
from pandas.io.pytables import Term
def test_query_long_float_literal(setup_path):
    df = DataFrame({'A': [1000000000.0009, 1000000000.0011, 1000000000.0015]})
    with ensure_clean_store(setup_path) as store:
        store.append('test', df, format='table', data_columns=True)
        cutoff = 1000000000.0006
        result = store.select('test', f'A < {cutoff:.4f}')
        assert result.empty
        cutoff = 1000000000.001
        result = store.select('test', f'A > {cutoff:.4f}')
        expected = df.loc[[1, 2], :]
        tm.assert_frame_equal(expected, result)
        exact = 1000000000.0011
        result = store.select('test', f'A == {exact:.4f}')
        expected = df.loc[[1], :]
        tm.assert_frame_equal(expected, result)