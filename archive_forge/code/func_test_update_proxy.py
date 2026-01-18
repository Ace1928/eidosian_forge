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
def test_update_proxy(self):
    """Verify that ``LazyProxyCategoricalDtype._update_proxy`` method works as expected."""
    lazy_proxy, _, _ = self._get_lazy_proxy()
    new_parent = pd.DataFrame({'a': [10, 20, 30]})._query_compiler._modin_frame
    assert isinstance(lazy_proxy, LazyProxyCategoricalDtype)
    assert lazy_proxy._update_proxy(lazy_proxy._parent, lazy_proxy._column_name) is lazy_proxy
    proxy_with_new_column = lazy_proxy._update_proxy(lazy_proxy._parent, 'other_column')
    assert proxy_with_new_column is not lazy_proxy and isinstance(proxy_with_new_column, LazyProxyCategoricalDtype)
    proxy_with_new_parent = lazy_proxy._update_proxy(new_parent, lazy_proxy._column_name)
    assert proxy_with_new_parent is not lazy_proxy and isinstance(proxy_with_new_parent, LazyProxyCategoricalDtype)
    lazy_proxy.categories
    assert type(lazy_proxy._update_proxy(lazy_proxy._parent, lazy_proxy._column_name)) == pandas.CategoricalDtype