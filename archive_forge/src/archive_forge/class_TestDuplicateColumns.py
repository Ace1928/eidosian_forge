import os
import re
import numpy as np
import pandas
import pyarrow
import pytest
from pandas._testing import ensure_clean
from pandas.core.dtypes.common import is_list_like
from pyhdk import __version__ as hdk_version
from modin.config import StorageFormat
from modin.tests.interchange.dataframe_protocol.hdk.utils import split_df_into_chunks
from modin.tests.pandas.utils import (
from .utils import ForceHdkImport, eval_io, run_and_compare, set_execution_mode
import modin.pandas as pd
from modin.experimental.core.execution.native.implementations.hdk_on_native.calcite_serializer import (
from modin.experimental.core.execution.native.implementations.hdk_on_native.df_algebra import (
from modin.experimental.core.execution.native.implementations.hdk_on_native.partitioning.partition_manager import (
from modin.pandas.io import from_arrow
from modin.tests.pandas.utils import (
from modin.utils import try_cast_to_pandas
class TestDuplicateColumns:

    def test_init(self):

        def init(df, **kwargs):
            return df
        data = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]]
        columns = ['c1', 'c2', 'c1', 'c3']
        run_and_compare(fn=init, data=data, force_lazy=False, constructor_kwargs={'columns': columns})

    def test_loc(self):

        def loc(df, **kwargs):
            return df.loc[:, ['col1', 'col3', 'col3']]
        run_and_compare(fn=loc, data=test_data_values[0], force_lazy=False)

    def test_set_columns(self):

        def set_cols(df, **kwargs):
            df.columns = ['col1', 'col3', 'col3']
            return df
        run_and_compare(fn=set_cols, data=[[1, 2, 3], [4, 5, 6], [7, 8, 9]], force_lazy=False)

    def test_set_axis(self):

        def set_axis(df, **kwargs):
            sort_index = df.axes[1]
            labels = [np.nan if i % 2 == 0 else sort_index[i] for i in range(len(sort_index))]
            return df.set_axis(labels, axis=1, copy=kwargs['copy'])
        run_and_compare(fn=set_axis, data=test_data['float_nan_data'], force_lazy=False, copy=True)
        run_and_compare(fn=set_axis, data=test_data['float_nan_data'], force_lazy=False, copy=False)