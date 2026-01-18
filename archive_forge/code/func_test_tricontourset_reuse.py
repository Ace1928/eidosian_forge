import numpy as np
from numpy.testing import (
import numpy.ma.testutils as matest
import pytest
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.path import Path
from matplotlib.testing.decorators import image_comparison, check_figures_equal
def test_tricontourset_reuse():
    x = [0.0, 0.5, 1.0]
    y = [0.0, 1.0, 0.0]
    z = [1.0, 2.0, 3.0]
    fig, ax = plt.subplots()
    tcs1 = ax.tricontourf(x, y, z)
    tcs2 = ax.tricontour(x, y, z)
    assert tcs2._contour_generator != tcs1._contour_generator
    tcs3 = ax.tricontour(tcs1, z)
    assert tcs3._contour_generator == tcs1._contour_generator