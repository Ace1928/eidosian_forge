import inspect
import numpy as np
import pandas
import pytest
import modin.pandas as pd
def test_series_str_api_equality():
    modin_dir = [obj for obj in dir(pd.Series.str) if obj[0] != '_']
    pandas_dir = [obj for obj in dir(pandas.Series.str) if obj[0] != '_']
    missing_from_modin = set(pandas_dir) - set(modin_dir)
    assert not len(missing_from_modin), 'Differences found in API: {}'.format(missing_from_modin)
    extra_in_modin = set(modin_dir) - set(pandas_dir)
    assert not len(extra_in_modin), 'Differences found in API: {}'.format(extra_in_modin)
    assert_parameters_eq((pandas.Series.str, pd.Series.str), modin_dir, [])