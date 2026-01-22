from copy import copy, deepcopy
import numpy as np
import pytest
import xarray as xr
import xarray.datatree_.datatree.testing as dtt
import xarray.testing as xrt
from xarray.core.datatree import DataTree
from xarray.core.treenode import NotFoundInTreeError
from xarray.tests import create_test_data, source_ndarray
class TestTreeCreation:

    def test_empty(self):
        dt: DataTree = DataTree(name='root')
        assert dt.name == 'root'
        assert dt.parent is None
        assert dt.children == {}
        xrt.assert_identical(dt.to_dataset(), xr.Dataset())

    def test_unnamed(self):
        dt: DataTree = DataTree()
        assert dt.name is None

    def test_bad_names(self):
        with pytest.raises(TypeError):
            DataTree(name=5)
        with pytest.raises(ValueError):
            DataTree(name='folder/data')