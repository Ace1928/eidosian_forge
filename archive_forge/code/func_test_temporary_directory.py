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
def test_temporary_directory(tmpdir):
    b = db.range(10, npartitions=4)
    with ProcessPoolExecutor(4) as pool:
        with dask.config.set(temporary_directory=str(tmpdir), pool=pool):
            b2 = b.groupby(lambda x: x % 2)
            b2.compute()
            assert any((fn.endswith('.partd') for fn in os.listdir(str(tmpdir))))