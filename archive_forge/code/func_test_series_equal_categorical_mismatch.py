import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_series_equal_categorical_mismatch(check_categorical, using_infer_string):
    if using_infer_string:
        dtype = 'string'
    else:
        dtype = 'object'
    msg = f"""Attributes of Series are different\n\nAttribute "dtype" are different\n\\[left\\]:  CategoricalDtype\\(categories=\\['a', 'b'\\], ordered=False, categories_dtype={dtype}\\)\n\\[right\\]: CategoricalDtype\\(categories=\\['a', 'b', 'c'\\], ordered=False, categories_dtype={dtype}\\)"""
    s1 = Series(Categorical(['a', 'b']))
    s2 = Series(Categorical(['a', 'b'], categories=list('abc')))
    if check_categorical:
        with pytest.raises(AssertionError, match=msg):
            tm.assert_series_equal(s1, s2, check_categorical=check_categorical)
    else:
        _assert_series_equal_both(s1, s2, check_categorical=check_categorical)