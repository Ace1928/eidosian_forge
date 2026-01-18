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
@mpl3d_image_comparison(['poly3dcollection_closed.png'], style='mpl20')
def test_poly3dcollection_closed():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    poly1 = np.array([[0, 0, 1], [0, 1, 1], [0, 0, 0]], float)
    poly2 = np.array([[0, 1, 1], [1, 1, 1], [1, 1, 0]], float)
    c1 = art3d.Poly3DCollection([poly1], linewidths=3, edgecolor='k', facecolor=(0.5, 0.5, 1, 0.5), closed=True)
    c2 = art3d.Poly3DCollection([poly2], linewidths=3, edgecolor='k', facecolor=(1, 0.5, 0.5, 0.5), closed=False)
    ax.add_collection3d(c1)
    ax.add_collection3d(c2)