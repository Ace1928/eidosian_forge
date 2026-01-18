from itertools import product
import io
import platform
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import cbook
from matplotlib.backend_bases import MouseEvent
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle, Ellipse
from matplotlib.transforms import Bbox, TransformedBbox
from matplotlib.testing.decorators import (
from mpl_toolkits.axes_grid1 import (
from mpl_toolkits.axes_grid1.anchored_artists import (
from mpl_toolkits.axes_grid1.axes_divider import (
from mpl_toolkits.axes_grid1.axes_rgb import RGBAxes
from mpl_toolkits.axes_grid1.inset_locator import (
import mpl_toolkits.axes_grid1.mpl_axes
import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
@pytest.mark.parametrize('click_on', ['big', 'small'])
@pytest.mark.parametrize('big_on_axes,small_on_axes', [('gca', 'gca'), ('host', 'host'), ('host', 'parasite'), ('parasite', 'host'), ('parasite', 'parasite')])
def test_picking_callbacks_overlap(big_on_axes, small_on_axes, click_on):
    """Test pick events on normal, host or parasite axes."""
    big = plt.Rectangle((0.25, 0.25), 0.5, 0.5, picker=5)
    small = plt.Rectangle((0.4, 0.4), 0.2, 0.2, facecolor='r', picker=5)
    received_events = []

    def on_pick(event):
        received_events.append(event)
    plt.gcf().canvas.mpl_connect('pick_event', on_pick)
    rectangles_on_axes = (big_on_axes, small_on_axes)
    axes = {'gca': None, 'host': None, 'parasite': None}
    if 'gca' in rectangles_on_axes:
        axes['gca'] = plt.gca()
    if 'host' in rectangles_on_axes or 'parasite' in rectangles_on_axes:
        axes['host'] = host_subplot(111)
        axes['parasite'] = axes['host'].twin()
    axes[big_on_axes].add_patch(big)
    axes[small_on_axes].add_patch(small)
    if click_on == 'big':
        click_axes = axes[big_on_axes]
        axes_coords = (0.3, 0.3)
    else:
        click_axes = axes[small_on_axes]
        axes_coords = (0.5, 0.5)
    if click_axes is axes['parasite']:
        click_axes = axes['host']
    x, y = click_axes.transAxes.transform(axes_coords)
    m = MouseEvent('button_press_event', click_axes.figure.canvas, x, y, button=1)
    click_axes.pick(m)
    expected_n_events = 2 if click_on == 'small' else 1
    assert len(received_events) == expected_n_events
    event_rects = [event.artist for event in received_events]
    assert big in event_rects
    if click_on == 'small':
        assert small in event_rects