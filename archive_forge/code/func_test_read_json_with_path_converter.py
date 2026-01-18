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
def test_read_json_with_path_converter():
    path_column_name = 'filenames'

    def path_converter(x):
        return 'asdf.json'
    with tmpfile('json') as f:
        df.to_json(f, orient='records', lines=False)
        actual = dd.read_json(f, orient='records', lines=False, include_path_column=path_column_name, path_converter=path_converter)
        actual_pd = pd.read_json(f, orient='records', lines=False)
        actual_pd[path_column_name] = pd.Series((path_converter(f),) * len(actual_pd), dtype='category')
        assert_eq(actual, actual_pd)