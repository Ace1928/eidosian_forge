from copy import copy, deepcopy
import numpy as np
import pytest
import xarray as xr
import xarray.datatree_.datatree.testing as dtt
import xarray.testing as xrt
from xarray.core.datatree import DataTree
from xarray.core.treenode import NotFoundInTreeError
from xarray.tests import create_test_data, source_ndarray
class TestTreeFromDict:

    def test_data_in_root(self):
        dat = xr.Dataset()
        dt = DataTree.from_dict({'/': dat})
        assert dt.name is None
        assert dt.parent is None
        assert dt.children == {}
        xrt.assert_identical(dt.to_dataset(), dat)

    def test_one_layer(self):
        dat1, dat2 = (xr.Dataset({'a': 1}), xr.Dataset({'b': 2}))
        dt = DataTree.from_dict({'run1': dat1, 'run2': dat2})
        xrt.assert_identical(dt.to_dataset(), xr.Dataset())
        assert dt.name is None
        xrt.assert_identical(dt['run1'].to_dataset(), dat1)
        assert dt['run1'].children == {}
        xrt.assert_identical(dt['run2'].to_dataset(), dat2)
        assert dt['run2'].children == {}

    def test_two_layers(self):
        dat1, dat2 = (xr.Dataset({'a': 1}), xr.Dataset({'a': [1, 2]}))
        dt = DataTree.from_dict({'highres/run': dat1, 'lowres/run': dat2})
        assert 'highres' in dt.children
        assert 'lowres' in dt.children
        highres_run = dt['highres/run']
        xrt.assert_identical(highres_run.to_dataset(), dat1)

    def test_nones(self):
        dt = DataTree.from_dict({'d': None, 'd/e': None})
        assert [node.name for node in dt.subtree] == [None, 'd', 'e']
        assert [node.path for node in dt.subtree] == ['/', '/d', '/d/e']
        xrt.assert_identical(dt['d/e'].to_dataset(), xr.Dataset())

    def test_full(self, simple_datatree):
        dt = simple_datatree
        paths = list((node.path for node in dt.subtree))
        assert paths == ['/', '/set1', '/set1/set1', '/set1/set2', '/set2', '/set2/set1', '/set3']

    def test_datatree_values(self):
        dat1: DataTree = DataTree(data=xr.Dataset({'a': 1}))
        expected: DataTree = DataTree()
        expected['a'] = dat1
        actual = DataTree.from_dict({'a': dat1})
        dtt.assert_identical(actual, expected)

    def test_roundtrip(self, simple_datatree):
        dt = simple_datatree
        roundtrip = DataTree.from_dict(dt.to_dict())
        assert roundtrip.equals(dt)

    @pytest.mark.xfail
    def test_roundtrip_unnamed_root(self, simple_datatree):
        dt = simple_datatree
        dt.name = 'root'
        roundtrip = DataTree.from_dict(dt.to_dict())
        assert roundtrip.equals(dt)