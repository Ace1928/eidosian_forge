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
def test_orc_names(orc_files, tmp_path):
    df = dd.read_orc(orc_files)
    assert df._name.startswith('read-orc')
    out = df.to_orc(tmp_path, compute=False)
    assert out._name.startswith('to-orc')