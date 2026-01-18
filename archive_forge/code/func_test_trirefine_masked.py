import numpy as np
from numpy.testing import (
import numpy.ma.testutils as matest
import pytest
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.path import Path
from matplotlib.testing.decorators import image_comparison, check_figures_equal
@pytest.mark.parametrize('interpolator', [mtri.LinearTriInterpolator, mtri.CubicTriInterpolator], ids=['linear', 'cubic'])
def test_trirefine_masked(interpolator):
    x, y = np.mgrid[:2, :2]
    x = np.repeat(x.flatten(), 2)
    y = np.repeat(y.flatten(), 2)
    z = np.zeros_like(x)
    tri = mtri.Triangulation(x, y)
    refiner = mtri.UniformTriRefiner(tri)
    interp = interpolator(tri, z)
    refiner.refine_field(z, triinterpolator=interp, subdiv=2)