from copy import copy, deepcopy
import numpy as np
import pytest
import xarray as xr
import xarray.datatree_.datatree.testing as dtt
import xarray.testing as xrt
from xarray.core.datatree import DataTree
from xarray.core.treenode import NotFoundInTreeError
from xarray.tests import create_test_data, source_ndarray
def test_set_data(self):
    john: DataTree = DataTree(name='john')
    dat = xr.Dataset({'a': 0})
    john.ds = dat
    xrt.assert_identical(john.to_dataset(), dat)
    with pytest.raises(TypeError):
        john.ds = 'junk'