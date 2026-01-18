from copy import copy, deepcopy
import numpy as np
import pytest
import xarray as xr
import xarray.datatree_.datatree.testing as dtt
import xarray.testing as xrt
from xarray.core.datatree import DataTree
from xarray.core.treenode import NotFoundInTreeError
from xarray.tests import create_test_data, source_ndarray
def test_relative_paths(self):
    sue: DataTree = DataTree()
    mary: DataTree = DataTree(children={'Sue': sue})
    annie: DataTree = DataTree()
    john: DataTree = DataTree(children={'Mary': mary, 'Annie': annie})
    result = sue.relative_to(john)
    assert result == 'Mary/Sue'
    assert john.relative_to(sue) == '../..'
    assert annie.relative_to(sue) == '../../Annie'
    assert sue.relative_to(annie) == '../Mary/Sue'
    assert sue.relative_to(sue) == '.'
    evil_kate: DataTree = DataTree()
    with pytest.raises(NotFoundInTreeError, match='nodes do not lie within the same tree'):
        sue.relative_to(evil_kate)