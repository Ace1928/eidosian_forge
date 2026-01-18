from copy import copy, deepcopy
import numpy as np
import pytest
import xarray as xr
import xarray.datatree_.datatree.testing as dtt
import xarray.testing as xrt
from xarray.core.datatree import DataTree
from xarray.core.treenode import NotFoundInTreeError
from xarray.tests import create_test_data, source_ndarray
def test_operation_with_attrs_but_no_data(self):
    xs = xr.Dataset({'testvar': xr.DataArray(np.ones((2, 3)))})
    dt = DataTree.from_dict({'node1': xs, 'node2': xs})
    dt.attrs['test_key'] = 1
    dt.sel(dim_0=0)