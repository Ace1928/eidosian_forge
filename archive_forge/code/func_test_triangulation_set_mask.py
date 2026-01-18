import numpy as np
from numpy.testing import (
import numpy.ma.testutils as matest
import pytest
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.path import Path
from matplotlib.testing.decorators import image_comparison, check_figures_equal
def test_triangulation_set_mask():
    x = [-1, 0, 1, 0]
    y = [0, -1, 0, 1]
    triangles = [[0, 1, 2], [2, 3, 0]]
    triang = mtri.Triangulation(x, y, triangles)
    assert_array_equal(triang.neighbors, [[-1, -1, 1], [-1, -1, 0]])
    triang.set_mask([False, True])
    assert_array_equal(triang.mask, [False, True])
    triang.set_mask(None)
    assert triang.mask is None
    msg = 'mask array must have same length as triangles array'
    for mask in ([False, True, False], [False], [True], False, True):
        with pytest.raises(ValueError, match=msg):
            triang.set_mask(mask)