import itertools
import platform
import timeit
from types import SimpleNamespace
from cycler import cycler
import numpy as np
from numpy.testing import assert_array_equal
import pytest
import matplotlib
import matplotlib as mpl
from matplotlib import _path
import matplotlib.lines as mlines
from matplotlib.markers import MarkerStyle
from matplotlib.path import Path
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.testing.decorators import image_comparison, check_figures_equal
def test_set_line_coll_dash():
    fig, ax = plt.subplots()
    np.random.seed(0)
    ax.contour(np.random.randn(20, 30), linestyles=[(0, (3, 3))])