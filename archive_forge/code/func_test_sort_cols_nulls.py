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
@pytest.mark.parametrize('ascending', ascending_values)
@pytest.mark.parametrize('na_position', na_position_values)
def test_sort_cols_nulls(self, cols, ascending, na_position):

    def sort(df, cols, ascending, na_position, **kwargs):
        return df.sort_values(cols, ascending=ascending, na_position=na_position)
    run_and_compare(sort, data=self.data_nulls, cols=cols, ascending=ascending, na_position=na_position)