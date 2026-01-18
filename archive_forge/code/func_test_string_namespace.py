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
def test_string_namespace():
    b = db.from_sequence(['Alice Smith', 'Bob Jones', 'Charlie Smith'], npartitions=2)
    assert 'split' in dir(b.str)
    assert 'match' in dir(b.str)
    assert list(b.str.lower()) == ['alice smith', 'bob jones', 'charlie smith']
    assert list(b.str.split(' ')) == [['Alice', 'Smith'], ['Bob', 'Jones'], ['Charlie', 'Smith']]
    assert list(b.str.match('*Smith')) == ['Alice Smith', 'Charlie Smith']
    pytest.raises(AttributeError, lambda: b.str.sfohsofhf)
    assert b.str.match('*Smith').name == b.str.match('*Smith').name
    assert b.str.match('*Smith').name != b.str.match('*John').name