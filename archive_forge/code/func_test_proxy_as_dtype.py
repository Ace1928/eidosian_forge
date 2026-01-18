import matplotlib
import numpy as np
import pandas
import pytest
import modin.pandas as pd
from modin.config import MinPartitionSize, NPartitions, StorageFormat
from modin.core.dataframe.pandas.metadata import LazyProxyCategoricalDtype
from modin.core.storage_formats.pandas.utils import split_result_of_axis_func_pandas
from modin.pandas.testing import assert_index_equal, assert_series_equal
from modin.tests.pandas.utils import (
from modin.tests.test_utils import warns_that_defaulting_to_pandas
from modin.utils import get_current_execution
def test_proxy_as_dtype(self):
    """Verify that proxy can be used as an actual dtype."""
    lazy_proxy, actual_dtype, _ = self._get_lazy_proxy()
    assert isinstance(lazy_proxy, LazyProxyCategoricalDtype)
    assert not lazy_proxy._is_materialized
    modin_df2, pandas_df2 = create_test_dfs({'c': [2, 2, 3, 4, 5, 6]})
    eval_general((modin_df2, lazy_proxy), (pandas_df2, actual_dtype), lambda args: args[0].astype({'c': args[1]}))