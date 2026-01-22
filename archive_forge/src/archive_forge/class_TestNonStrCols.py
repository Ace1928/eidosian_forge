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
class TestNonStrCols:
    data = {0: [1, 2, 3], '1': [3, 4, 5], 2: [6, 7, 8]}

    def test_sum(self):
        mdf = pd.DataFrame(self.data).sum()
        pdf = pandas.DataFrame(self.data).sum()
        df_equals(mdf, pdf)

    def test_set_index(self):
        df = pd.DataFrame(self.data)
        df._query_compiler._modin_frame._set_index(pd.Index([1, 2, 3]))