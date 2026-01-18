from __future__ import annotations
import pytest
from dask.context import config
from os import path
from itertools import product
import datashader as ds
import xarray as xr
import numpy as np
import dask.array as da
import pandas as pd
from datashader.resampling import compute_chunksize
import datashader.transfer_functions as tf
from packaging.version import Version
def test_resample_compute_chunksize():
    """
    Ensure chunksize computation is correct.
    """
    darr = da.from_array(np.zeros((100, 100)), (10, 10))
    mem_limited_chunksize = compute_chunksize(darr, 10, 10, max_mem=2000)
    assert mem_limited_chunksize == (2, 1)
    explicit_chunksize = compute_chunksize(darr, 10, 10, chunksize=(5, 4))
    assert explicit_chunksize == (5, 4)