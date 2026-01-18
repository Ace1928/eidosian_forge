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
def test_to_textfiles_endlines():
    b = db.from_sequence(['a', 'b', 'c'], npartitions=1)
    with tmpfile() as fn:
        for last_endline in (False, True):
            b.to_textfiles([fn], last_endline=last_endline)
            with open(fn) as f:
                result = f.readlines()
            assert result == ['a\n', 'b\n', 'c\n' if last_endline else 'c']