import numpy as np
import pytest
from pandas._libs.tslibs import Timestamp
import pandas as pd
from pandas import (
from pandas.tests.io.pytables.common import (
from pandas.io.pytables import Term
def test_query_with_nested_special_character(setup_path):
    df = DataFrame({'a': ['a', 'a', 'c', 'b', 'test & test', 'c', 'b', 'e'], 'b': [1, 2, 3, 4, 5, 6, 7, 8]})
    expected = df[df.a == 'test & test']
    with ensure_clean_store(setup_path) as store:
        store.append('test', df, format='table', data_columns=True)
        result = store.select('test', 'a = "test & test"')
    tm.assert_frame_equal(expected, result)