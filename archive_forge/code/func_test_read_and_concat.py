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
def test_read_and_concat(self):
    csv_file = os.path.join(self.root, 'modin/tests/pandas/data', 'test_usecols.csv')
    ref1 = pandas.read_csv(csv_file)
    ref2 = pandas.read_csv(csv_file)
    ref = pandas.concat([ref1, ref2])
    exp1 = pandas.read_csv(csv_file)
    exp2 = pandas.read_csv(csv_file)
    exp = pd.concat([exp1, exp2])
    with ForceHdkImport(exp):
        df_equals(ref, exp)