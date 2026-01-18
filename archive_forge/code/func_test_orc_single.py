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
@pytest.mark.parametrize('split_stripes', [1, 2])
@pytest.mark.network
def test_orc_single(orc_files, split_stripes):
    fn = orc_files[0]
    d = dd.read_orc(fn, split_stripes=split_stripes)
    assert len(d) == 70000
    assert d.npartitions == 8 / split_stripes
    d2 = dd.read_orc(fn, columns=['time', 'date'])
    assert_eq(d[columns], d2[columns], check_index=False)
    with pytest.raises(ValueError, match='nonexist'):
        dd.read_orc(fn, columns=['time', 'nonexist'])
    if not dd._dask_expr_enabled():
        d3 = d[columns]
        keys = [(d3._name, i) for i in range(d3.npartitions)]
        graph = optimize_dataframe_getitem(d3.__dask_graph__(), keys)
        key = [k for k in graph.layers.keys() if k.startswith('read-orc-')][0]
        assert set(graph.layers[key].columns) == set(columns)