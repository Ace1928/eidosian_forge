import numpy as np
from numpy.testing import (
import numpy.ma.testutils as matest
import pytest
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.path import Path
from matplotlib.testing.decorators import image_comparison, check_figures_equal
def test_qhull_large_offset():
    x = np.asarray([0, 1, 0, 1, 0.5])
    y = np.asarray([0, 0, 1, 1, 0.5])
    offset = 10000000000.0
    triang = mtri.Triangulation(x, y)
    triang_offset = mtri.Triangulation(x + offset, y + offset)
    assert len(triang.triangles) == len(triang_offset.triangles)