import numpy as np
from numpy.testing import (
import numpy.ma.testutils as matest
import pytest
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.path import Path
from matplotlib.testing.decorators import image_comparison, check_figures_equal
def test_triplot_return():
    ax = plt.figure().add_subplot()
    triang = mtri.Triangulation([0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0], triangles=[[0, 1, 3], [3, 2, 0]])
    assert ax.triplot(triang, 'b-') is not None, 'triplot should return the artist it adds'