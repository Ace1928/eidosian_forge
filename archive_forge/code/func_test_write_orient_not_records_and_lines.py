from __future__ import annotations
import json
import os
import fsspec
import pandas as pd
import pytest
from packaging.version import Version
import dask
import dask.dataframe as dd
from dask.dataframe._compat import PANDAS_GE_200
from dask.dataframe.utils import assert_eq
from dask.utils import tmpdir, tmpfile
def test_write_orient_not_records_and_lines():
    with tmpfile('json') as f:
        with pytest.raises(ValueError, match='Line-delimited JSON'):
            dd.to_json(ddf, f, orient='split', lines=True)