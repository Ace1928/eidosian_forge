import matplotlib
import numpy as np
import pandas
import pytest
import modin.pandas as pd
from modin.config import NPartitions, StorageFormat
from modin.tests.pandas.utils import (
@pytest.mark.parametrize('axis', ['rows', 'columns'])
@pytest.mark.parametrize('ddof', int_arg_values, ids=arg_keys('ddof', int_arg_keys))
@pytest.mark.parametrize('method', ['std', 'var'])
def test_std_var_transposed(axis, ddof, method):
    eval_general(*create_test_dfs(test_data['int_data']), lambda df: getattr(df.T, method)(axis=axis, ddof=ddof))