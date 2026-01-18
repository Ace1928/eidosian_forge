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
def test_usecols_csv(self):
    """check with the following arguments: names, dtype, skiprows, delimiter"""
    csv_file = os.path.join(self.root, 'modin/tests/pandas/data', 'test_usecols.csv')
    for kwargs in ({'delimiter': ','}, {'sep': None}, {'skiprows': 1, 'names': ['A', 'B', 'C', 'D', 'E']}, {'dtype': {'a': 'int32', 'e': 'string'}}, {'dtype': {'a': np.dtype('int32'), 'b': np.dtype('int64'), 'e': 'string'}}):
        eval_io(fn_name='read_csv', md_extra_kwargs={'engine': 'arrow'}, filepath_or_buffer=csv_file, **kwargs)