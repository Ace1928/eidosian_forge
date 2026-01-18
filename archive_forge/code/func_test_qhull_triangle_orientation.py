import numpy as np
from numpy.testing import (
import numpy.ma.testutils as matest
import pytest
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.path import Path
from matplotlib.testing.decorators import image_comparison, check_figures_equal
def test_qhull_triangle_orientation():
    xi = np.linspace(-2, 2, 100)
    x, y = map(np.ravel, np.meshgrid(xi, xi))
    w = (x > y - 1) & (x < -1.95) & (y > -1.2)
    x, y = (x[w], y[w])
    theta = np.radians(25)
    x1 = x * np.cos(theta) - y * np.sin(theta)
    y1 = x * np.sin(theta) + y * np.cos(theta)
    triang = mtri.Triangulation(x1, y1)
    qhull_neighbors = triang.neighbors
    triang._neighbors = None
    own_neighbors = triang.neighbors
    assert_array_equal(qhull_neighbors, own_neighbors)