from copy import copy, deepcopy
import numpy as np
import pytest
import xarray as xr
import xarray.datatree_.datatree.testing as dtt
import xarray.testing as xrt
from xarray.core.datatree import DataTree
from xarray.core.treenode import NotFoundInTreeError
from xarray.tests import create_test_data, source_ndarray
def test_update_doesnt_alter_child_name(self):
    dt: DataTree = DataTree()
    dt.update({'foo': xr.DataArray(0), 'a': DataTree(name='b')})
    assert 'a' in dt.children
    child = dt['a']
    assert child.name == 'a'