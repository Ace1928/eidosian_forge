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
@pytest.mark.parametrize('use_data_coordinates', [False, True])
def test_rectangle_resize_square_center_aspect(ax, use_data_coordinates):
    ax.set_aspect(0.8)
    tool = widgets.RectangleSelector(ax, onselect=noop, interactive=True, use_data_coordinates=use_data_coordinates)
    click_and_drag(tool, start=(70, 65), end=(120, 115))
    assert tool.extents == (70.0, 120.0, 65.0, 115.0)
    tool.add_state('square')
    tool.add_state('center')
    if use_data_coordinates:
        extents = tool.extents
        xdata, ydata, width = (extents[1], extents[3], extents[1] - extents[0])
        xdiff, ycenter = (10, extents[2] + (extents[3] - extents[2]) / 2)
        xdata_new, ydata_new = (xdata + xdiff, ydata)
        ychange = width / 2 + xdiff
        click_and_drag(tool, start=(xdata, ydata), end=(xdata_new, ydata_new))
        assert_allclose(tool.extents, [extents[0] - xdiff, xdata_new, ycenter - ychange, ycenter + ychange])
    else:
        extents = tool.extents
        xdata, ydata = (extents[1], extents[3])
        xdiff = 10
        xdata_new, ydata_new = (xdata + xdiff, ydata)
        ychange = xdiff * 1 / tool._aspect_ratio_correction
        click_and_drag(tool, start=(xdata, ydata), end=(xdata_new, ydata_new))
        assert_allclose(tool.extents, [extents[0] - xdiff, xdata_new, 46.25, 133.75])