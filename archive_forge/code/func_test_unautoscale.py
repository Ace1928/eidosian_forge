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
@pytest.mark.parametrize('axis', ('x', 'y', 'z'))
@pytest.mark.parametrize('auto', (True, False, None))
def test_unautoscale(axis, auto):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    x = np.arange(100)
    y = np.linspace(-0.1, 0.1, 100)
    ax.scatter(x, y)
    get_autoscale_on = getattr(ax, f'get_autoscale{axis}_on')
    set_lim = getattr(ax, f'set_{axis}lim')
    get_lim = getattr(ax, f'get_{axis}lim')
    post_auto = get_autoscale_on() if auto is None else auto
    set_lim((-0.5, 0.5), auto=auto)
    assert post_auto == get_autoscale_on()
    fig.canvas.draw()
    np.testing.assert_array_equal(get_lim(), (-0.5, 0.5))