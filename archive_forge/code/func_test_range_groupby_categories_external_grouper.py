import datetime
import itertools
from unittest import mock
import numpy as np
import pandas
import pandas._libs.lib as lib
import pytest
import modin.pandas as pd
from modin.config import (
from modin.core.dataframe.algebra.default2pandas.groupby import GroupBy
from modin.core.dataframe.pandas.partitioning.axis_partition import (
from modin.pandas.io import from_pandas
from modin.pandas.utils import is_scalar
from modin.tests.test_utils import warns_that_defaulting_to_pandas
from modin.utils import (
from .utils import (
@pytest.mark.parametrize('cat_cols', [['a'], ['b'], ['a', 'b']])
@pytest.mark.parametrize('columns', [[(False, 'a'), (True, 'b')], [(True, 'a')], [(True, 'a'), (True, 'b')]])
def test_range_groupby_categories_external_grouper(columns, cat_cols):
    data = {'a': [1, 1, 2, 2] * 64, 'b': [11, 11, 22, 22] * 64, 'c': [111, 111, 222, 222] * 64, 'data': [1, 2, 3, 4] * 64}
    md_df, pd_df = create_test_dfs(data)
    md_df = md_df.astype({col: 'category' for col in cat_cols})
    pd_df = pd_df.astype({col: 'category' for col in cat_cols})
    md_df, md_by = get_external_groupers(md_df, columns, drop_from_original_df=True)
    pd_df, pd_by = get_external_groupers(pd_df, columns, drop_from_original_df=True)
    eval_general(md_df.groupby(md_by), pd_df.groupby(pd_by), lambda grp: grp.count())