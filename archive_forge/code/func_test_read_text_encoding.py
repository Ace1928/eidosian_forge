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
def test_read_text_encoding():
    with tmpfile() as fn:
        with open(fn, 'wb') as f:
            f.write(('你好！' + os.linesep).encode('gb18030') * 100)
        b = db.read_text(fn, blocksize=100, encoding='gb18030')
        c = db.read_text(fn, encoding='gb18030')
        assert len(b.dask) > 5
        b_enc = b.str.strip().map(lambda x: x.encode('utf-8'))
        c_enc = c.str.strip().map(lambda x: x.encode('utf-8'))
        assert list(b_enc) == list(c_enc)
        d = db.read_text([fn], blocksize=100, encoding='gb18030')
        assert list(b) == list(d)