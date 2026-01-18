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
def test_orc_with_backend():
    pytest.importorskip('requests')
    d = dd.read_orc(url)
    assert set(d.columns) == {'time', 'date'}
    assert len(d) == 70000