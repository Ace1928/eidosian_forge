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
def test_singleton_autolim():
    fig, ax = plt.subplots()
    ax.scatter(0, 0)
    np.testing.assert_allclose(ax.get_ylim(), [-0.06, 0.06])
    np.testing.assert_allclose(ax.get_xlim(), [-0.06, 0.06])