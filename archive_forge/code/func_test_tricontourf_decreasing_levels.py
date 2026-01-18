import numpy as np
from numpy.testing import (
import numpy.ma.testutils as matest
import pytest
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.path import Path
from matplotlib.testing.decorators import image_comparison, check_figures_equal
def test_tricontourf_decreasing_levels():
    x = [0.0, 1.0, 1.0]
    y = [0.0, 0.0, 1.0]
    z = [0.2, 0.4, 0.6]
    plt.figure()
    with pytest.raises(ValueError):
        plt.tricontourf(x, y, z, [1.0, 0.0])