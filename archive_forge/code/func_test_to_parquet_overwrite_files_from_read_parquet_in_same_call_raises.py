from __future__ import annotations
import contextlib
import glob
import math
import os
import sys
import warnings
from datetime import date
from decimal import Decimal
from unittest.mock import MagicMock
import numpy as np
import pandas as pd
import pytest
from packaging.version import parse as parse_version
import dask
import dask.dataframe as dd
import dask.multiprocessing
from dask.array.numpy_compat import NUMPY_GE_124
from dask.blockwise import Blockwise, optimize_blockwise
from dask.dataframe._compat import (
from dask.dataframe.io.parquet.core import get_engine
from dask.dataframe.io.parquet.utils import _parse_pandas_metadata
from dask.dataframe.optimize import optimize_dataframe_getitem
from dask.dataframe.utils import assert_eq, pyarrow_strings_enabled
from dask.layers import DataFrameIOLayer
from dask.utils import natural_sort_key
from dask.utils_test import hlg_layer
@pytest.mark.xfail(sys.platform == 'win32' and DASK_EXPR_ENABLED, reason='File not found error on windows')
def test_to_parquet_overwrite_files_from_read_parquet_in_same_call_raises(tmpdir, engine):
    subdir = tmpdir.mkdir('subdir')
    dd.from_pandas(pd.DataFrame({'x': range(20)}), npartitions=2).to_parquet(subdir, engine=engine)
    ddf = dd.read_parquet(subdir)
    for target in [subdir, tmpdir]:
        with pytest.raises(ValueError, match='same parquet file|Cannot overwrite a path'):
            ddf.to_parquet(target, overwrite=True)
        ddf2 = ddf.assign(y=ddf.x + 1)
        with pytest.raises(ValueError, match='same parquet file|Cannot overwrite a path'):
            ddf2.to_parquet(target, overwrite=True)