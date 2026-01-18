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
@pytest.mark.parametrize('md_df_constructor', [pytest.param(pd.DataFrame, id='from_pandas_dataframe'), pytest.param(lambda pd_df: from_arrow(pyarrow.Table.from_pandas(pd_df)), id='from_pyarrow_table')])
def test_uint_overflow(self, md_df_constructor):
    """
        Verify that the exception is arisen when overflow occurs due to 'uint -> int' compatibility conversion.

        Originally, HDK does not support unsigned integers, there's a logic in Modin that upcasts
        unsigned types to the compatible ones prior importing to HDK. This test ensures that the
        error is arisen when such conversion causes a data loss.
        """
    md_df = md_df_constructor(pandas.DataFrame({'col': np.array([2 ** 64 - 1, 2 ** 64 - 2, 2 ** 64 - 3], dtype='uint64')}))
    with pytest.raises(OverflowError):
        with ForceHdkImport(md_df):
            pass