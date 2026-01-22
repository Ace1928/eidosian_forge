from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray import DataArray, Dataset
from xarray.core.datatree import DataTree
from xarray.tests import create_test_data, requires_dask
def _create_test_datatree(modify=lambda ds: ds):
    set1_data = modify(xr.Dataset({'a': 0, 'b': 1}))
    set2_data = modify(xr.Dataset({'a': ('x', [2, 3]), 'b': ('x', [0.1, 0.2])}))
    root_data = modify(xr.Dataset({'a': ('y', [6, 7, 8]), 'set0': ('x', [9, 10])}))
    root: DataTree = DataTree(data=root_data)
    set1: DataTree = DataTree(name='set1', parent=root, data=set1_data)
    DataTree(name='set1', parent=set1)
    DataTree(name='set2', parent=set1)
    set2: DataTree = DataTree(name='set2', parent=root, data=set2_data)
    DataTree(name='set1', parent=set2)
    DataTree(name='set3', parent=root)
    return root