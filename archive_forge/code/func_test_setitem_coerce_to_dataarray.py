from copy import copy, deepcopy
import numpy as np
import pytest
import xarray as xr
import xarray.datatree_.datatree.testing as dtt
import xarray.testing as xrt
from xarray.core.datatree import DataTree
from xarray.core.treenode import NotFoundInTreeError
from xarray.tests import create_test_data, source_ndarray
def test_setitem_coerce_to_dataarray(self):
    folder1: DataTree = DataTree(name='folder1')
    folder1['results'] = 0
    xrt.assert_equal(folder1['results'], xr.DataArray(0))