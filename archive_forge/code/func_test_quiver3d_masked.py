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
@mpl3d_image_comparison(['quiver3d_masked.png'], style='mpl20')
def test_quiver3d_masked():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    x, y, z = np.mgrid[-1:0.8:10j, -1:0.8:10j, -1:0.6:3j]
    u = np.sin(np.pi * x) * np.cos(np.pi * y) * np.cos(np.pi * z)
    v = -np.cos(np.pi * x) * np.sin(np.pi * y) * np.cos(np.pi * z)
    w = (2 / 3) ** 0.5 * np.cos(np.pi * x) * np.cos(np.pi * y) * np.sin(np.pi * z)
    u = np.ma.masked_where((-0.4 < x) & (x < 0.1), u, copy=False)
    v = np.ma.masked_where((0.1 < y) & (y < 0.7), v, copy=False)
    ax.quiver(x, y, z, u, v, w, length=0.1, pivot='tip', normalize=True)