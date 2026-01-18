import numpy as np
from numpy.testing import (
import numpy.ma.testutils as matest
import pytest
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.path import Path
from matplotlib.testing.decorators import image_comparison, check_figures_equal
def tri_contains_point(xtri, ytri, xy):
    tri_points = np.vstack((xtri, ytri)).T
    return Path(tri_points).contains_point(xy)