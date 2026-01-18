from collections import abc
import email
from email.parser import Parser
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_to_records_with_categorical(self):
    df = DataFrame({'A': list('abc')}, dtype='category')
    expected = Series(list('abc'), dtype='category', name='A')
    tm.assert_series_equal(df['A'], expected)
    df = DataFrame(list('abc'), dtype='category')
    expected = Series(list('abc'), dtype='category', name=0)
    tm.assert_series_equal(df[0], expected)
    result = df.to_records()
    expected = np.rec.array([(0, 'a'), (1, 'b'), (2, 'c')], dtype=[('index', '=i8'), ('0', 'O')])
    tm.assert_almost_equal(result, expected)