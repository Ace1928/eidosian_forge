from __future__ import annotations
import contextlib
import itertools
import multiprocessing as mp
import os
import pickle
import random
import string
import tempfile
from concurrent.futures import ProcessPoolExecutor
from copy import copy
from datetime import date, time
from decimal import Decimal
from functools import partial
from unittest import mock
import numpy as np
import pandas as pd
import pytest
import dask
import dask.dataframe as dd
from dask.base import compute_as_if_collection
from dask.dataframe._compat import (
from dask.dataframe.shuffle import (
from dask.dataframe.utils import assert_eq, make_meta
from dask.optimization import cull
@pytest.mark.skipif(DASK_EXPR_ENABLED, reason='not valid for dask-expr')
@pytest.mark.parametrize('npartitions', [10, 1])
def test_shuffle_hlg_layer_serialize(npartitions):
    ddf = dd.from_pandas(pd.DataFrame({'a': np.random.randint(0, 10, 100)}), npartitions=npartitions)
    ddf_shuffled = ddf.shuffle('a', max_branch=3, shuffle_method='tasks')
    dsk = ddf_shuffled.__dask_graph__()
    for layer in dsk.layers.values():
        if not isinstance(layer, dd.shuffle.SimpleShuffleLayer):
            continue
        assert not hasattr(layer, '_cached_dict')
        layer_roundtrip = pickle.loads(pickle.dumps(layer))
        assert type(layer_roundtrip) == type(layer)
        assert not hasattr(layer_roundtrip, '_cached_dict')
        assert layer_roundtrip.keys() == layer.keys()