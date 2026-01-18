import warnings
import matplotlib
import numpy as np
import pandas
import pytest
import modin.pandas as pd
from modin.config import Engine, NPartitions, RangePartitioning, StorageFormat
from modin.pandas.io import to_pandas
from modin.tests.pandas.utils import (
from modin.tests.test_utils import warns_that_defaulting_to_pandas
@pytest.mark.parametrize('sort_remaining', bool_arg_values, ids=arg_keys('sort_remaining', bool_arg_keys))
def test_sort_multiindex(sort_remaining):
    data = test_data['int_data']
    modin_df, pandas_df = (pd.DataFrame(data), pandas.DataFrame(data))
    for index in ['index', 'columns']:
        new_index = generate_multiindex(len(getattr(modin_df, index)))
        for df in [modin_df, pandas_df]:
            setattr(df, index, new_index)
    for kwargs in [{'level': 0}, {'axis': 0}, {'axis': 1}]:
        with warns_that_defaulting_to_pandas():
            df_equals(modin_df.sort_index(sort_remaining=sort_remaining, **kwargs), pandas_df.sort_index(sort_remaining=sort_remaining, **kwargs))