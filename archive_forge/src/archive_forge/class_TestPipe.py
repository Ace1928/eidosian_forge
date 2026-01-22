from copy import copy, deepcopy
import numpy as np
import pytest
import xarray as xr
import xarray.datatree_.datatree.testing as dtt
import xarray.testing as xrt
from xarray.core.datatree import DataTree
from xarray.core.treenode import NotFoundInTreeError
from xarray.tests import create_test_data, source_ndarray
class TestPipe:

    def test_noop(self, create_test_datatree):
        dt = create_test_datatree()
        actual = dt.pipe(lambda tree: tree)
        assert actual.identical(dt)

    def test_params(self, create_test_datatree):
        dt = create_test_datatree()

        def f(tree, **attrs):
            return tree.assign(arr_with_attrs=xr.Variable('dim0', [], attrs=attrs))
        attrs = {'x': 1, 'y': 2, 'z': 3}
        actual = dt.pipe(f, **attrs)
        assert actual['arr_with_attrs'].attrs == attrs

    def test_named_self(self, create_test_datatree):
        dt = create_test_datatree()

        def f(x, tree, y):
            tree.attrs.update({'x': x, 'y': y})
            return tree
        attrs = {'x': 1, 'y': 2}
        actual = dt.pipe((f, 'tree'), **attrs)
        assert actual is dt and actual.attrs == attrs