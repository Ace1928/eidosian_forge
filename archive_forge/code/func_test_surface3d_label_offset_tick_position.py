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
@image_comparison(['surface3d_label_offset_tick_position.png'], style='mpl20')
def test_surface3d_label_offset_tick_position():
    ax = plt.figure().add_subplot(projection='3d')
    x, y = np.mgrid[0:6 * np.pi:0.25, 0:4 * np.pi:0.25]
    z = np.sqrt(np.abs(np.cos(x) + np.cos(y)))
    ax.plot_surface(x * 100000.0, y * 1000000.0, z * 100000000.0, cmap='autumn', cstride=2, rstride=2)
    ax.set_xlabel('X label')
    ax.set_ylabel('Y label')
    ax.set_zlabel('Z label')
    ax.figure.canvas.draw()