import io
import warnings
import matplotlib
import numpy as np
import pandas
import pandas._libs.lib as lib
import pytest
from numpy.testing import assert_array_equal
import modin.pandas as pd
from modin.config import Engine, NPartitions, StorageFormat
from modin.pandas.io import to_pandas
from modin.tests.pandas.utils import (
from modin.tests.test_utils import warns_that_defaulting_to_pandas
from modin.utils import get_current_execution
@pytest.mark.parametrize('aggfunc', [pytest.param('sum', id='MapReduce_func'), pytest.param('nunique', id='FullAxis_func')])
@pytest.mark.parametrize('margins', [True, False])
def test_pivot_table_fill_value(aggfunc, margins):
    md_df, pd_df = create_test_dfs(test_data['int_data'])
    eval_general(md_df, pd_df, operation=lambda df, *args, **kwargs: df.pivot_table(*args, **kwargs), index=md_df.columns[0], columns=md_df.columns[1], values=md_df.columns[2], aggfunc=aggfunc, margins=margins, fill_value=10)