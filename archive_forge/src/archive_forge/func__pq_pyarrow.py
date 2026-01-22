from __future__ import annotations
import os
import pytest
import sys
from operator import getitem
from distributed import Client, SchedulerPlugin
from distributed.utils_test import cluster, loop  # noqa F401
from dask.highlevelgraph import HighLevelGraph
from dask.layers import ArrayChunkShapeDep, ArraySliceDep, fractional_slice
def _pq_pyarrow(tmpdir):
    pytest.importorskip('pyarrow')
    pd = pytest.importorskip('pandas')
    dd = pytest.importorskip('dask.dataframe')
    dd.from_pandas(pd.DataFrame({'a': range(10)}), npartitions=2).to_parquet(str(tmpdir))
    filters = [('a', '<=', 2)]
    ddf1 = dd.read_parquet(str(tmpdir), filters=filters)
    return ddf1