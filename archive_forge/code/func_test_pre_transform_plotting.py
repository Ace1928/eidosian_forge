import copy
import numpy as np
from numpy.testing import (assert_allclose, assert_almost_equal,
import pytest
from matplotlib import scale
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
from matplotlib.transforms import Affine2D, Bbox, TransformedBbox
from matplotlib.path import Path
from matplotlib.testing.decorators import image_comparison, check_figures_equal
@image_comparison(['pre_transform_data'], remove_text=True, style='mpl20', tol=0.05)
def test_pre_transform_plotting():
    ax = plt.axes()
    times10 = mtransforms.Affine2D().scale(10)
    ax.contourf(np.arange(48).reshape(6, 8), transform=times10 + ax.transData)
    ax.pcolormesh(np.linspace(0, 4, 7), np.linspace(5.5, 8, 9), np.arange(48).reshape(8, 6), transform=times10 + ax.transData)
    ax.scatter(np.linspace(0, 10), np.linspace(10, 0), transform=times10 + ax.transData)
    x = np.linspace(8, 10, 20)
    y = np.linspace(1, 5, 20)
    u = 2 * np.sin(x) + np.cos(y[:, np.newaxis])
    v = np.sin(x) - np.cos(y[:, np.newaxis])
    ax.streamplot(x, y, u, v, transform=times10 + ax.transData, linewidth=np.hypot(u, v))
    x, y = (x[::3], y[::3])
    u, v = (u[::3, ::3], v[::3, ::3])
    ax.quiver(x, y + 5, u, v, transform=times10 + ax.transData)
    ax.barbs(x - 3, y + 5, u ** 2, v ** 2, transform=times10 + ax.transData)