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
@mpl3d_image_comparison(['surface3d_zsort_inf.png'], style='mpl20')
def test_surface3d_zsort_inf():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    x, y = np.mgrid[-2:2:0.1, -2:2:0.1]
    z = np.sin(x) ** 2 + np.cos(y) ** 2
    z[x.shape[0] // 2:, x.shape[1] // 2:] = np.inf
    ax.plot_surface(x, y, z, cmap='jet')
    ax.view_init(elev=45, azim=145)