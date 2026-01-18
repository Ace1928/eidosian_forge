from copy import copy, deepcopy
import numpy as np
import pytest
import xarray as xr
import xarray.datatree_.datatree.testing as dtt
import xarray.testing as xrt
from xarray.core.datatree import DataTree
from xarray.core.treenode import NotFoundInTreeError
from xarray.tests import create_test_data, source_ndarray
def test_setitem_unnamed_child_node_becomes_named(self):
    john2: DataTree = DataTree(name='john2')
    john2['sonny'] = DataTree()
    assert john2['sonny'].name == 'sonny'