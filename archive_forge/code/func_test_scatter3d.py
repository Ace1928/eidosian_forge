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
@mpl3d_image_comparison(['scatter3d.png'], style='mpl20')
def test_scatter3d():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(np.arange(10), np.arange(10), np.arange(10), c='r', marker='o')
    x = y = z = np.arange(10, 20)
    ax.scatter(x, y, z, c='b', marker='^')
    z[-1] = 0
    ax.scatter([], [], [], c='r', marker='X')