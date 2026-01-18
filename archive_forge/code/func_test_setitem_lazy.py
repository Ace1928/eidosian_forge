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
def test_setitem_lazy(self):

    def applier(df, **kwargs):
        df = df + 1
        df['a'] = df['a'] + 1
        df['e'] = df['a'] + 1
        df['new_int8'] = np.int8(10)
        df['new_int16'] = np.int16(10)
        df['new_int32'] = np.int32(10)
        df['new_int64'] = np.int64(10)
        df['new_int'] = 10
        df['new_float'] = 5.5
        df['new_float64'] = np.float64(10.1)
        return df
    run_and_compare(applier, data=self.data)