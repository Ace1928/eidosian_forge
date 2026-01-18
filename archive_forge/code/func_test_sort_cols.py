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
@pytest.mark.parametrize('cols', cols_values)
@pytest.mark.parametrize('ignore_index', bool_arg_values)
@pytest.mark.parametrize('ascending', ascending_values)
@pytest.mark.parametrize('index_cols', index_cols_values)
def test_sort_cols(self, cols, ignore_index, index_cols, ascending):

    def sort(df, cols, ignore_index, index_cols, ascending, **kwargs):
        if index_cols:
            df = df.set_index(index_cols)
            df_equals_with_non_stable_indices()
        return df.sort_values(cols, ignore_index=ignore_index, ascending=ascending)
    run_and_compare(sort, data=self.data, cols=cols, ignore_index=ignore_index, index_cols=index_cols, ascending=ascending, force_lazy=index_cols is None)