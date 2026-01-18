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
@pytest.mark.parametrize('tool,button,key,expected', [('zoom', MouseButton.LEFT, None, ((0.0, 0.06), (0.01, 0.07), (0.02, 0.08))), ('zoom', MouseButton.LEFT, 'x', ((-0.01, 0.1), (-0.03, 0.08), (-0.06, 0.06))), ('zoom', MouseButton.LEFT, 'y', ((-0.07, 0.04), (-0.03, 0.08), (0.0, 0.11))), ('zoom', MouseButton.RIGHT, None, ((-0.09, 0.15), (-0.07, 0.17), (-0.06, 0.18))), ('pan', MouseButton.LEFT, None, ((-0.7, -0.58), (-1.03, -0.91), (-1.27, -1.15))), ('pan', MouseButton.LEFT, 'x', ((-0.96, -0.84), (-0.58, -0.46), (-0.06, 0.06))), ('pan', MouseButton.LEFT, 'y', ((0.2, 0.32), (-0.51, -0.39), (-1.27, -1.15)))])
def test_toolbar_zoom_pan(tool, button, key, expected):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(0, 0, 0)
    fig.canvas.draw()
    xlim0, ylim0, zlim0 = (ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d())
    d0 = (0, 0)
    d1 = (1, 1)
    s0 = ax.transData.transform(d0).astype(int)
    s1 = ax.transData.transform(d1).astype(int)
    start_event = MouseEvent('button_press_event', fig.canvas, *s0, button, key=key)
    stop_event = MouseEvent('button_release_event', fig.canvas, *s1, button, key=key)
    tb = NavigationToolbar2(fig.canvas)
    if tool == 'zoom':
        tb.zoom()
        tb.press_zoom(start_event)
        tb.drag_zoom(stop_event)
        tb.release_zoom(stop_event)
    else:
        tb.pan()
        tb.press_pan(start_event)
        tb.drag_pan(stop_event)
        tb.release_pan(stop_event)
    xlim, ylim, zlim = expected
    assert ax.get_xlim3d() == pytest.approx(xlim, abs=0.01)
    assert ax.get_ylim3d() == pytest.approx(ylim, abs=0.01)
    assert ax.get_zlim3d() == pytest.approx(zlim, abs=0.01)
    tb.back()
    assert ax.get_xlim3d() == pytest.approx(xlim0)
    assert ax.get_ylim3d() == pytest.approx(ylim0)
    assert ax.get_zlim3d() == pytest.approx(zlim0)
    tb.forward()
    assert ax.get_xlim3d() == pytest.approx(xlim, abs=0.01)
    assert ax.get_ylim3d() == pytest.approx(ylim, abs=0.01)
    assert ax.get_zlim3d() == pytest.approx(zlim, abs=0.01)
    tb.home()
    assert ax.get_xlim3d() == pytest.approx(xlim0)
    assert ax.get_ylim3d() == pytest.approx(ylim0)
    assert ax.get_zlim3d() == pytest.approx(zlim0)