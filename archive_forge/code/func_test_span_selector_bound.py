import functools
import io
from unittest import mock
import matplotlib as mpl
from matplotlib.backend_bases import MouseEvent
import matplotlib.colors as mcolors
import matplotlib.widgets as widgets
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.testing.decorators import check_figures_equal, image_comparison
from matplotlib.testing.widgets import (click_and_drag, do_event, get_ax,
import numpy as np
from numpy.testing import assert_allclose
import pytest
@pytest.mark.parametrize('direction', ('horizontal', 'vertical'))
def test_span_selector_bound(direction):
    fig, ax = plt.subplots(1, 1)
    ax.plot([10, 20], [10, 30])
    ax.figure.canvas.draw()
    x_bound = ax.get_xbound()
    y_bound = ax.get_ybound()
    tool = widgets.SpanSelector(ax, print, direction, interactive=True)
    assert ax.get_xbound() == x_bound
    assert ax.get_ybound() == y_bound
    bound = x_bound if direction == 'horizontal' else y_bound
    assert tool._edge_handles.positions == list(bound)
    press_data = (10.5, 11.5)
    move_data = (11, 13)
    release_data = move_data
    click_and_drag(tool, start=press_data, end=move_data)
    assert ax.get_xbound() == x_bound
    assert ax.get_ybound() == y_bound
    index = 0 if direction == 'horizontal' else 1
    handle_positions = [press_data[index], release_data[index]]
    assert tool._edge_handles.positions == handle_positions