from __future__ import annotations
from collections.abc import Hashable
import numpy as np
import pandas as pd
import pytest
from xarray.core import duck_array_ops, utils
from xarray.core.utils import either_dict_or_kwargs, infix_dims, iterate_nested
from xarray.tests import assert_array_equal, requires_dask
def test_repr_object_magic_methods():
    o1 = utils.ReprObject('foo')
    o2 = utils.ReprObject('foo')
    o3 = utils.ReprObject('bar')
    o4 = 'foo'
    assert o1 == o2
    assert o1 != o3
    assert o1 != o4
    assert hash(o1) == hash(o2)
    assert hash(o1) != hash(o3)
    assert hash(o1) != hash(o4)