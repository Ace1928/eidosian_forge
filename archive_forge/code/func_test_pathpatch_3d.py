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
def test_pathpatch_3d(fig_test, fig_ref):
    ax = fig_ref.add_subplot(projection='3d')
    path = Path.unit_rectangle()
    patch = PathPatch(path)
    art3d.pathpatch_2d_to_3d(patch, z=(0, 0.5, 0.7, 1, 0), zdir='y')
    ax.add_artist(patch)
    ax = fig_test.add_subplot(projection='3d')
    pp3d = art3d.PathPatch3D(path, zs=(0, 0.5, 0.7, 1, 0), zdir='y')
    ax.add_artist(pp3d)