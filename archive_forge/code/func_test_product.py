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
def test_product():
    b2 = b.product(b)
    assert b2.npartitions == b.npartitions ** 2
    assert set(b2) == {(i, j) for i in L for j in L}
    x = db.from_sequence([1, 2, 3, 4])
    y = db.from_sequence([10, 20, 30])
    z = x.product(y)
    assert set(z) == {(i, j) for i in [1, 2, 3, 4] for j in [10, 20, 30]}
    assert z.name != b2.name
    assert z.name == x.product(y).name