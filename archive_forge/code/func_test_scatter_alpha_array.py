from datetime import datetime
import io
import itertools
import re
from types import SimpleNamespace
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
import pytest
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.collections as mcollections
import matplotlib.colors as mcolors
import matplotlib.path as mpath
import matplotlib.transforms as mtransforms
from matplotlib.collections import (Collection, LineCollection,
from matplotlib.testing.decorators import check_figures_equal, image_comparison
def test_scatter_alpha_array():
    x = np.arange(5)
    alpha = x / 5
    fig, (ax0, ax1) = plt.subplots(2)
    sc0 = ax0.scatter(x, x, c=x, alpha=alpha)
    sc1 = ax1.scatter(x, x, c=x)
    sc1.set_alpha(alpha)
    plt.draw()
    assert_array_equal(sc0.get_facecolors()[:, -1], alpha)
    assert_array_equal(sc1.get_facecolors()[:, -1], alpha)
    fig, (ax0, ax1) = plt.subplots(2)
    sc0 = ax0.scatter(x, x, color=['r', 'g', 'b', 'c', 'm'], alpha=alpha)
    sc1 = ax1.scatter(x, x, color='r', alpha=alpha)
    plt.draw()
    assert_array_equal(sc0.get_facecolors()[:, -1], alpha)
    assert_array_equal(sc1.get_facecolors()[:, -1], alpha)
    fig, (ax0, ax1) = plt.subplots(2)
    sc0 = ax0.scatter(x, x, color=['r', 'g', 'b', 'c', 'm'])
    sc0.set_alpha(alpha)
    sc1 = ax1.scatter(x, x, color='r')
    sc1.set_alpha(alpha)
    plt.draw()
    assert_array_equal(sc0.get_facecolors()[:, -1], alpha)
    assert_array_equal(sc1.get_facecolors()[:, -1], alpha)