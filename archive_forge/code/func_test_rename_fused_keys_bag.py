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
def test_rename_fused_keys_bag():
    inp = {'b': (list, 'a'), 'c': (f, 'b', 1)}
    outp = optimize(inp, ['c'], rename_fused_keys=False)
    assert outp.keys() == {'c'}
    assert outp['c'][1:] == ('a', 1)
    with dask.config.set({'optimization.fuse.rename-keys': False}):
        assert optimize(inp, ['c']) == outp
    assert optimize(inp, ['c']) != outp