import functools
import itertools
import platform
import pytest
from mpl_toolkits.mplot3d import Axes3D, axes3d, proj3d, art3d
import matplotlib as mpl
from matplotlib.backend_bases import (MouseButton, MouseEvent,
from matplotlib import cm
from matplotlib import colors as mcolors, patches as mpatch
from matplotlib.testing.decorators import image_comparison, check_figures_equal
from matplotlib.testing.widgets import mock_event
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.patches import Circle, PathPatch
from matplotlib.path import Path
from matplotlib.text import Text
import matplotlib.pyplot as plt
import numpy as np
@check_figures_equal(extensions=['png'])
def test_scatter3d_linewidth_modification(fig_ref, fig_test):
    ax_test = fig_test.add_subplot(projection='3d')
    c = ax_test.scatter(np.arange(10), np.arange(10), np.arange(10), marker='o')
    c.set_linewidths(np.arange(10))
    ax_ref = fig_ref.add_subplot(projection='3d')
    ax_ref.scatter(np.arange(10), np.arange(10), np.arange(10), marker='o', linewidths=np.arange(10))