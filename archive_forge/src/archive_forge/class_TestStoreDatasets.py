from copy import copy, deepcopy
import numpy as np
import pytest
import xarray as xr
import xarray.datatree_.datatree.testing as dtt
import xarray.testing as xrt
from xarray.core.datatree import DataTree
from xarray.core.treenode import NotFoundInTreeError
from xarray.tests import create_test_data, source_ndarray
class TestStoreDatasets:

    def test_create_with_data(self):
        dat = xr.Dataset({'a': 0})
        john: DataTree = DataTree(name='john', data=dat)
        xrt.assert_identical(john.to_dataset(), dat)
        with pytest.raises(TypeError):
            DataTree(name='mary', parent=john, data='junk')

    def test_set_data(self):
        john: DataTree = DataTree(name='john')
        dat = xr.Dataset({'a': 0})
        john.ds = dat
        xrt.assert_identical(john.to_dataset(), dat)
        with pytest.raises(TypeError):
            john.ds = 'junk'

    def test_has_data(self):
        john: DataTree = DataTree(name='john', data=xr.Dataset({'a': 0}))
        assert john.has_data
        john_no_data: DataTree = DataTree(name='john', data=None)
        assert not john_no_data.has_data

    def test_is_hollow(self):
        john: DataTree = DataTree(data=xr.Dataset({'a': 0}))
        assert john.is_hollow
        eve: DataTree = DataTree(children={'john': john})
        assert eve.is_hollow
        eve.ds = xr.Dataset({'a': 1})
        assert not eve.is_hollow