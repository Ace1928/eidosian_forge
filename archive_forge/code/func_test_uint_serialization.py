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
def test_uint_serialization(self):
    df = pd.DataFrame({'A': [np.nan, 1]})
    assert df.fillna(np.uint8(np.iinfo(np.uint8).max)).sum()[0] == np.iinfo(np.uint8).max + 1
    assert df.fillna(np.uint16(np.iinfo(np.uint16).max)).sum()[0] == np.iinfo(np.uint16).max + 1
    assert df.fillna(np.uint32(np.iinfo(np.uint32).max)).sum()[0] == np.iinfo(np.uint32).max + 1
    assert df.fillna(np.uint64(np.iinfo(np.int64).max - 1)).sum()[0] == np.iinfo(np.int64).max
    df = pd.DataFrame({'A': [np.iinfo(np.uint8).max, 1]})
    assert df.astype(np.uint8).sum()[0] == np.iinfo(np.uint8).max + 1
    df = pd.DataFrame({'A': [np.iinfo(np.uint16).max, 1]})
    assert df.astype(np.uint16).sum()[0] == np.iinfo(np.uint16).max + 1
    df = pd.DataFrame({'A': [np.iinfo(np.uint32).max, 1]})
    assert df.astype(np.uint32).sum()[0] == np.iinfo(np.uint32).max + 1
    df = pd.DataFrame({'A': [np.iinfo(np.int64).max - 1, 1]})
    assert df.astype(np.uint64).sum()[0] == np.iinfo(np.int64).max