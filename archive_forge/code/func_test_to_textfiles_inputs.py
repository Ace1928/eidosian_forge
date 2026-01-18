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
def test_to_textfiles_inputs():
    B = db.from_sequence(['abc', '123', 'xyz'], npartitions=2)
    with tmpfile() as a:
        with tmpfile() as b:
            B.to_textfiles([a, b])
            assert os.path.exists(a)
            assert os.path.exists(b)
    with tmpdir() as dirname:
        B.to_textfiles(dirname)
        assert os.path.exists(dirname)
        assert os.path.exists(os.path.join(dirname, '0.part'))
    with pytest.raises(TypeError):
        B.to_textfiles(5)