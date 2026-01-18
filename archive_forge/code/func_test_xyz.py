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
@mpl3d_image_comparison(['voxels-xyz.png'], tol=0.01, remove_text=False, style='mpl20')
def test_xyz(self):
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

    def midpoints(x):
        sl = ()
        for i in range(x.ndim):
            x = (x[sl + np.index_exp[:-1]] + x[sl + np.index_exp[1:]]) / 2.0
            sl += np.index_exp[:]
        return x
    r, g, b = np.indices((17, 17, 17)) / 16.0
    rc = midpoints(r)
    gc = midpoints(g)
    bc = midpoints(b)
    sphere = (rc - 0.5) ** 2 + (gc - 0.5) ** 2 + (bc - 0.5) ** 2 < 0.5 ** 2
    colors = np.zeros(sphere.shape + (3,))
    colors[..., 0] = rc
    colors[..., 1] = gc
    colors[..., 2] = bc
    ax.voxels(r, g, b, sphere, facecolors=colors, edgecolors=np.clip(2 * colors - 0.5, 0, 1), linewidth=0.5)