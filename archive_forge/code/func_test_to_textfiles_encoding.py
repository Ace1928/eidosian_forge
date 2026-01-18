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
def test_to_textfiles_encoding():
    b = db.from_sequence(['汽车', '苹果', '天气'], npartitions=2)
    for ext, myopen in ext_open:
        with tmpdir() as dir:
            c = b.to_textfiles(os.path.join(dir, '*.' + ext), encoding='gb18030', compute=False)
            dask.compute(*c)
            assert os.path.exists(os.path.join(dir, '1.' + ext))
            f = myopen(os.path.join(dir, '1.' + ext), 'rb')
            text = f.read()
            if hasattr(text, 'decode'):
                text = text.decode('gb18030')
            assert '天气' in text
            f.close()