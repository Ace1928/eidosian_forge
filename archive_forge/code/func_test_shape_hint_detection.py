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
@pytest.mark.parametrize('index', [None, pandas.Index([1, 2, 3]), pandas.MultiIndex.from_tuples([(1, 1), (2, 2), (3, 3)])])
def test_shape_hint_detection(self, index):
    df = pd.DataFrame({'a': [1, 2, 3]}, index=index)
    assert df._query_compiler._shape_hint == 'column'
    transposed_data = df._to_pandas().T.to_dict()
    df = pd.DataFrame(transposed_data)
    assert df._query_compiler._shape_hint == 'row'
    df = pd.DataFrame({'a': [1, 2, 3], 'b': [1, 2, 3]}, index=index)
    assert df._query_compiler._shape_hint is None
    df = pd.DataFrame({'a': [1]}, index=None if index is None else index[:1])
    assert df._query_compiler._shape_hint == 'column'