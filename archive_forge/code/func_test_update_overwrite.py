from copy import copy, deepcopy
import numpy as np
import pytest
import xarray as xr
import xarray.datatree_.datatree.testing as dtt
import xarray.testing as xrt
from xarray.core.datatree import DataTree
from xarray.core.treenode import NotFoundInTreeError
from xarray.tests import create_test_data, source_ndarray
def test_update_overwrite(self):
    actual = DataTree.from_dict({'a': DataTree(xr.Dataset({'x': 1}))})
    actual.update({'a': DataTree(xr.Dataset({'x': 2}))})
    expected = DataTree.from_dict({'a': DataTree(xr.Dataset({'x': 2}))})
    print(actual)
    print(expected)
    dtt.assert_equal(actual, expected)