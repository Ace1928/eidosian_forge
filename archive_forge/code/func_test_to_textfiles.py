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
@pytest.mark.parametrize('ext,myopen', ext_open)
def test_to_textfiles(ext, myopen):
    b = db.from_sequence(['abc', '123', 'xyz'], npartitions=2)
    with tmpdir() as dir:
        c = b.to_textfiles(os.path.join(dir, '*.' + ext), compute=False)
        dask.compute(*c, scheduler='sync')
        assert os.path.exists(os.path.join(dir, '1.' + ext))
        f = myopen(os.path.join(dir, '1.' + ext), 'rb')
        text = f.read()
        if hasattr(text, 'decode'):
            text = text.decode()
        assert 'xyz' in text
        f.close()