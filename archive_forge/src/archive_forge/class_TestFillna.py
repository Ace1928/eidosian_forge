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
class TestFillna:
    data = {'a': [1, 1, None], 'b': [None, None, 2], 'c': [3, None, None]}
    values = [1, {'a': 1, 'c': 3}, {'a': 1, 'b': 2, 'c': 3}]

    @pytest.mark.parametrize('value', values)
    def test_fillna_all(self, value):

        def fillna(df, value, **kwargs):
            return df.fillna(value)
        run_and_compare(fillna, data=self.data, value=value)

    def test_fillna_bool(self):

        def fillna(df, **kwargs):
            df['a'] = df['a'] == 1
            df['a'] = df['a'].fillna(False)
            return df
        run_and_compare(fillna, data=self.data)