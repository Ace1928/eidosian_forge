import inspect
import numpy as np
import pandas
import pytest
import modin.pandas as pd
def test_series_dt_api_equality():
    modin_dir = [obj for obj in dir(pd.Series.dt) if obj[0] != '_']
    pandas_dir = [obj for obj in dir(pandas.Series.dt) if obj[0] != '_']
    ignore = ['week', 'weekofyear']
    missing_from_modin = set(pandas_dir) - set(modin_dir) - set(ignore)
    assert not len(missing_from_modin), 'Differences found in API: {}'.format(missing_from_modin)
    extra_in_modin = set(modin_dir) - set(pandas_dir)
    assert not len(extra_in_modin), 'Differences found in API: {}'.format(extra_in_modin)
    assert_parameters_eq((pandas.Series.dt, pd.Series.dt), modin_dir, [])