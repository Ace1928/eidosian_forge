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
@pytest.mark.parametrize('method', ['sum', 'mean', 'max', 'min', 'count', 'nunique'])
def test_simple_agg_no_default(self, method):

    def applier(df, **kwargs):
        if isinstance(df, pd.DataFrame):
            with pytest.warns(UserWarning) as warns:
                res = getattr(df, method)()
            for warn in warns.list:
                message = warn.message.args[0]
                if 'is_sparse is deprecated' in message or 'Frame contain columns with unsupported data-types' in message or 'Passing a BlockManager to DataFrame is deprecated' in message:
                    continue
                assert re.match('.*transpose.*defaulting to pandas', message) is not None, f'Expected DataFrame.transpose defaulting to pandas warning, got: {message}'
        else:
            res = getattr(df, method)()
        return res
    run_and_compare(applier, data=self.data, force_lazy=False)