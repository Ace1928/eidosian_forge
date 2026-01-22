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
@staticmethod
def _get_lazy_proxy():
    """
        Build a dataframe containing a column that has a proxy type and return
        this proxy together with an original dtype that this proxy is emulating.

        Returns
        -------
        (LazyProxyCategoricalDtype, pandas.CategoricalDtype, modin.pandas.DataFrame)
        """
    nchunks = 3
    pandas_df = pandas.DataFrame({'a': [1, 1, 2, 2, 3, 2], 'b': [1, 2, 3, 4, 5, 6]})
    original_dtype = pandas_df.astype({'a': 'category'}).dtypes['a']
    chunks = split_result_of_axis_func_pandas(axis=0, num_splits=nchunks, result=pandas_df, min_block_size=MinPartitionSize.get(), length_list=[2, 2, 2])
    if StorageFormat.get() == 'Pandas':
        df = pd.concat([pd.DataFrame(chunk) for chunk in chunks])
        assert df._query_compiler._modin_frame._partitions.shape == (nchunks, 1)
        df = df.astype({'a': 'category'})
        return (df.dtypes['a'], original_dtype, df)
    elif StorageFormat.get() == 'Hdk':
        import pyarrow as pa
        from modin.pandas.io import from_arrow
        at = pa.concat_tables([pa.Table.from_pandas(chunk.astype({'a': 'category'})) for chunk in chunks])
        assert len(at.column(0).chunks) == nchunks
        df = from_arrow(at)
        return (df.dtypes['a'], original_dtype, df)
    else:
        raise NotImplementedError()