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
def test_rectangle_handles(ax):
    tool = widgets.RectangleSelector(ax, onselect=noop, grab_range=10, interactive=True, handle_props={'markerfacecolor': 'r', 'markeredgecolor': 'b'})
    tool.extents = (100, 150, 100, 150)
    assert_allclose(tool.corners, ((100, 150, 150, 100), (100, 100, 150, 150)))
    assert tool.extents == (100, 150, 100, 150)
    assert_allclose(tool.edge_centers, ((100, 125.0, 150, 125.0), (125.0, 100, 125.0, 150)))
    assert tool.extents == (100, 150, 100, 150)
    click_and_drag(tool, start=(100, 100), end=(120, 120))
    assert tool.extents == (120, 150, 120, 150)
    click_and_drag(tool, start=(132, 132), end=(120, 120))
    assert tool.extents == (108, 138, 108, 138)
    click_and_drag(tool, start=(10, 10), end=(100, 100))
    assert tool.extents == (10, 100, 10, 100)
    assert mcolors.same_color(tool._corner_handles.artists[0].get_markerfacecolor(), 'r')
    assert mcolors.same_color(tool._corner_handles.artists[0].get_markeredgecolor(), 'b')