import ctypes
import math
import pytest
import modin.pandas as pd
@pytest.mark.parametrize('test_data', [{'a': ['foo', 'bar'], 'b': ['baz', 'qux']}, {'a': [1.5, 2.5, 3.5], 'b': [9.2, 10.5, 11.8]}, {'A': [1, 2, 3, 4], 'B': [1, 2, 3, 4]}], ids=['str_data', 'float_data', 'int_data'])
def test_only_one_dtype(test_data, df_from_dict):
    columns = list(test_data.keys())
    df = df_from_dict(test_data)
    dfX = df.__dataframe__()
    column_size = len(test_data[columns[0]])
    for column in columns:
        assert dfX.get_column_by_name(column).null_count == 0
        assert dfX.get_column_by_name(column).size() == column_size
        assert dfX.get_column_by_name(column).offset == 0