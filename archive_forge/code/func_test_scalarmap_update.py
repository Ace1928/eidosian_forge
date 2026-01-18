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
@mpl.style.context('default')
@check_figures_equal(extensions=['png'])
def test_scalarmap_update(fig_test, fig_ref):
    x, y, z = np.array(list(itertools.product(*[np.arange(0, 5, 1), np.arange(0, 5, 1), np.arange(0, 5, 1)]))).T
    c = x + y
    ax_test = fig_test.add_subplot(111, projection='3d')
    sc_test = ax_test.scatter(x, y, z, c=c, s=40, cmap='viridis')
    fig_test.canvas.draw()
    sc_test.changed()
    ax_ref = fig_ref.add_subplot(111, projection='3d')
    sc_ref = ax_ref.scatter(x, y, z, c=c, s=40, cmap='viridis')