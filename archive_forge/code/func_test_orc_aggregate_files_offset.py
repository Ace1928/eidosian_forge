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
@pytest.mark.network
def test_orc_aggregate_files_offset(orc_files):
    df2 = dd.read_orc(orc_files[:2], split_stripes=11, aggregate_files=True)
    assert df2.npartitions == 2
    assert len(df2.partitions[0].index) > len(df2.index) // 2