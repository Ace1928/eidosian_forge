from __future__ import annotations
import random
import sys
from copy import deepcopy
from itertools import product
import numpy as np
import pytest
import dask.array as da
from dask.array.numpy_compat import NUMPY_GE_123, ComplexWarning
from dask.array.utils import assert_eq
from dask.base import tokenize
from dask.utils import typename
@pytest.mark.parametrize('dtype', ('i8', 'f8'))
@pytest.mark.parametrize('reduction', ['sum', 'prod', 'mean', 'var', 'std', 'min', 'max', 'any', 'all'])
def test_reductions_allmasked(dtype, reduction):
    x = np.ma.masked_array([1, 2], dtype=dtype, mask=True)
    dx = da.from_array(x, asarray=False)
    dfunc = getattr(da, reduction)
    func = getattr(np, reduction)
    assert_eq_ma(dfunc(dx), func(x))