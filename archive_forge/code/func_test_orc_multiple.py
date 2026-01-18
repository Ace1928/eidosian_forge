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
def test_orc_multiple(orc_files):
    d = dd.read_orc(orc_files[0])
    d2 = dd.read_orc(orc_files)
    assert_eq(d2[columns], dd.concat([d, d])[columns], check_index=False)
    d2 = dd.read_orc(os.path.dirname(orc_files[0]) + '/*.orc')
    assert_eq(d2[columns], dd.concat([d, d])[columns], check_index=False)