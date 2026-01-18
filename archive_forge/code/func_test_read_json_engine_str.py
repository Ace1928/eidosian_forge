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
@pytest.mark.parametrize('engine', ['ujson', pd.read_json])
def test_read_json_engine_str(engine):
    with tmpfile('json') as f:
        df.to_json(f, lines=False)
        if isinstance(engine, str) and (not PANDAS_GE_200):
            with pytest.raises(ValueError, match='Pandas>=2.0 is required'):
                dd.read_json(f, engine=engine, lines=False)
        else:
            got = dd.read_json(f, engine=engine, lines=False)
            assert_eq(got, df)