from copy import copy, deepcopy
import numpy as np
import pytest
import xarray as xr
import xarray.datatree_.datatree.testing as dtt
import xarray.testing as xrt
from xarray.core.datatree import DataTree
from xarray.core.treenode import NotFoundInTreeError
from xarray.tests import create_test_data, source_ndarray
def test_update_new_named_dataarray(self):
    da = xr.DataArray(name='temp', data=[0, 50])
    folder1: DataTree = DataTree(name='folder1')
    folder1.update({'results': da})
    expected = da.rename('results')
    xrt.assert_equal(folder1['results'], expected)