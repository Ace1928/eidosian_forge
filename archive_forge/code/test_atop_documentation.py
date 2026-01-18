from __future__ import annotations
import collections
from operator import add
import numpy as np
import pytest
import dask
import dask.array as da
from dask.array.utils import assert_eq
from dask.blockwise import (
from dask.highlevelgraph import HighLevelGraph
from dask.utils_test import dec, hlg_layer_topological, inc

    Can we fuse blockwise layers with different, but compatible
    annotations for retries, priority, etc.
    