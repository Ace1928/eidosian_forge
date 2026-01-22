from copy import copy, deepcopy
import numpy as np
import pytest
import xarray as xr
import xarray.datatree_.datatree.testing as dtt
import xarray.testing as xrt
from xarray.core.datatree import DataTree
from xarray.core.treenode import NotFoundInTreeError
from xarray.tests import create_test_data, source_ndarray
class TestAccess:

    def test_attribute_access(self, create_test_datatree):
        dt = create_test_datatree()
        for key in ['a', 'set0']:
            xrt.assert_equal(dt[key], getattr(dt, key))
            assert key in dir(dt)
        xrt.assert_equal(dt['a']['y'], getattr(dt.a, 'y'))
        assert 'y' in dir(dt['a'])
        for key in ['set1', 'set2', 'set3']:
            dtt.assert_equal(dt[key], getattr(dt, key))
            assert key in dir(dt)
        dt.attrs['meta'] = 'NASA'
        assert dt.attrs['meta'] == 'NASA'
        assert 'meta' in dir(dt)

    def test_ipython_key_completions(self, create_test_datatree):
        dt = create_test_datatree()
        key_completions = dt._ipython_key_completions_()
        node_keys = [node.path[1:] for node in dt.subtree]
        assert all((node_key in key_completions for node_key in node_keys))
        var_keys = list(dt.variables.keys())
        assert all((var_key in key_completions for var_key in var_keys))

    def test_operation_with_attrs_but_no_data(self):
        xs = xr.Dataset({'testvar': xr.DataArray(np.ones((2, 3)))})
        dt = DataTree.from_dict({'node1': xs, 'node2': xs})
        dt.attrs['test_key'] = 1
        dt.sel(dim_0=0)