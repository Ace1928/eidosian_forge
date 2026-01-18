from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
import pytest
from packaging.version import Version
from pandas.api.types import is_object_dtype
import dask.dataframe as dd
from dask._compatibility import PY_VERSION
from dask.base import compute_as_if_collection
from dask.dataframe._compat import (
from dask.dataframe.core import _Frame
from dask.dataframe.methods import concat
from dask.dataframe.multi import (
from dask.dataframe.utils import (
from dask.utils_test import hlg_layer, hlg_layer_topological
@pytest.mark.parametrize('how', ['right', 'inner'])
def test_pairwise_rejects_unsupported_join_types(how):
    base_df = dd.from_pandas(pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]}, index=[0, 1, 3]), 3)
    dfs = [dd.from_pandas(pd.DataFrame({'a': [4, 5, 6], 'b': [3, 2, 1]}, index=[5, 6, 8]), 3), dd.from_pandas(pd.DataFrame({'a': [7, 8, 9], 'b': [0, 0, 0]}, index=[9, 9, 9]), 3)]
    with pytest.raises(ValueError) as e:
        base_df.join(dfs, how=how)
    e.match('merge_multi only supports left or outer joins')