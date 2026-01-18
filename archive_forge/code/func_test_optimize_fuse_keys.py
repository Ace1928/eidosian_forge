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
def test_optimize_fuse_keys():
    x = db.range(10, npartitions=2)
    y = x.map(inc)
    z = y.map(inc)
    dsk = z.__dask_optimize__(z.dask, z.__dask_keys__())
    assert not y.dask.keys() & dsk.keys()
    dsk = z.__dask_optimize__(z.dask, z.__dask_keys__(), fuse_keys=y.__dask_keys__())
    assert all((k in dsk for k in y.__dask_keys__()))