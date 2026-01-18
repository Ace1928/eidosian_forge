from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from pandas.tests.extension.decimal.array import DecimalDtype
from dask.dataframe._compat import PANDAS_GE_140, PANDAS_GE_150
from dask.dataframe._pyarrow import (
@pytest.mark.parametrize('series,expected', [(pd.DataFrame({'x': ['a', 'b']}, dtype=object), True), (pd.DataFrame({'x': ['a', 'b']}, dtype='string[python]'), True), (pd.DataFrame({'x': ['a', 'b']}, dtype='string[pyarrow]'), False), (pd.DataFrame({'x': [1, 2]}, dtype=int), False), (pd.DataFrame({'x': [1, 2]}, dtype=float), False), (pd.DataFrame({'x': [1, 2]}, dtype=float, index=pd.Index(['a', 'b'], dtype=object)), True), (pd.DataFrame({'x': [1, 2]}, dtype=float, index=pd.Index(['a', 'b'], dtype='string[pyarrow]')), False if PANDAS_GE_140 else True), (pd.Series({'x': ['a', 'b']}, dtype=object), False), (pd.Index({'x': ['a', 'b']}, dtype=object), False), (pd.MultiIndex.from_arrays([pd.Index(['a', 'a'], dtype=object), pd.Index(['a', 'b'], dtype=object)]), False), (pd.MultiIndex.from_arrays([pd.Index(['a', 'a'], dtype='string[python]'), pd.Index(['a', 'b'], dtype='string[pyarrow]')]), False), (pd.MultiIndex.from_arrays([pd.Index(['a', 'a'], dtype=object), pd.Index(['a', 'b'], dtype='string[pyarrow]')]), False)])
def tests_is_object_string_dataframe(series, expected):
    assert is_object_string_dataframe(series) is expected