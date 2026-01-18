from __future__ import annotations
import gc
import math
import os
import random
import warnings
import weakref
from bz2 import BZ2File
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from gzip import GzipFile
from itertools import repeat
import partd
import pytest
from tlz import groupby, identity, join, merge, pluck, unique, valmap
import dask
import dask.bag as db
from dask.bag.core import (
from dask.bag.utils import assert_eq
from dask.blockwise import Blockwise
from dask.delayed import Delayed
from dask.typing import Graph
from dask.utils import filetexts, tmpdir, tmpfile
from dask.utils_test import add, hlg_layer, hlg_layer_topological, inc
def test_repartition_partition_size_complex_dtypes():
    np = pytest.importorskip('numpy')
    b = db.from_sequence([np.array(range(100)) for _ in range(4)], npartitions=1)
    total_mem = sum(b.map_partitions(total_mem_usage).compute())
    new_partition_size = total_mem // 4
    c = b.repartition(partition_size=new_partition_size)
    assert c.npartitions >= 4
    assert_eq(b, c)