from copy import copy, deepcopy
import numpy as np
import pytest
import xarray as xr
import xarray.datatree_.datatree.testing as dtt
import xarray.testing as xrt
from xarray.core.datatree import DataTree
from xarray.core.treenode import NotFoundInTreeError
from xarray.tests import create_test_data, source_ndarray
def test_setitem_new_empty_node(self):
    john: DataTree = DataTree(name='john')
    john['mary'] = DataTree()
    mary = john['mary']
    assert isinstance(mary, DataTree)
    xrt.assert_identical(mary.to_dataset(), xr.Dataset())