from __future__ import annotations
import os
import warnings
from contextlib import nullcontext as does_not_warn
from itertools import permutations, zip_longest
import pytest
import itertools
import dask.array as da
import dask.config as config
from dask.array.numpy_compat import NUMPY_GE_122, ComplexWarning
from dask.array.utils import assert_eq, same_keys
from dask.core import get_deps
def test_reduction_names():
    x = da.ones(5, chunks=(2,))
    assert x.sum().name.startswith('sum')
    assert 'max' in x.max().name.split('-')[0]
    assert x.var().name.startswith('var')
    assert x.all().name.startswith('all')
    assert any((k[0].startswith('nansum') for k in da.nansum(x).dask))
    assert x.mean().name.startswith('mean')