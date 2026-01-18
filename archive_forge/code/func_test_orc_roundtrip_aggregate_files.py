from __future__ import annotations
import glob
import os
import shutil
import tempfile
import numpy as np
import pandas as pd
import pytest
import dask.dataframe as dd
from dask.dataframe.optimize import optimize_dataframe_getitem
from dask.dataframe.utils import assert_eq
@pytest.mark.parametrize('split_stripes', [True, False, 2, 4])
def test_orc_roundtrip_aggregate_files(tmpdir, split_stripes):
    tmp = str(tmpdir)
    data = pd.DataFrame({'a': np.arange(100, dtype=np.float64), 'b': np.random.choice(['cat', 'dog', 'mouse'], size=100)})
    df = dd.from_pandas(data, npartitions=8)
    df.to_orc(tmp, write_index=False)
    df2 = dd.read_orc(tmp, split_stripes=split_stripes, aggregate_files=True)
    if split_stripes:
        assert df2.npartitions == df.npartitions / int(split_stripes)
    else:
        assert df2.npartitions == df.npartitions
    assert_eq(data, df2, check_index=False)