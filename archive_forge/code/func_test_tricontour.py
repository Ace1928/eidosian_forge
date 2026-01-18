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
@mpl3d_image_comparison(['tricontour.png'], tol=0.02, style='mpl20')
def test_tricontour():
    fig = plt.figure()
    np.random.seed(19680801)
    x = np.random.rand(1000) - 0.5
    y = np.random.rand(1000) - 0.5
    z = -(x ** 2 + y ** 2)
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.tricontour(x, y, z)
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.tricontourf(x, y, z)