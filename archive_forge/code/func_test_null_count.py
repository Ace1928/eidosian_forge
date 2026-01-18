import ctypes
import math
import pytest
import modin.pandas as pd
def test_null_count(df_from_dict):
    df = df_from_dict({'foo': [42]})
    dfX = df.__dataframe__()
    colX = dfX.get_column_by_name('foo')
    null_count = colX.null_count
    assert null_count == 0 and type(null_count) is int