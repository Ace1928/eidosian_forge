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
@pytest.mark.parametrize('spancoords', ['data', 'pixels'])
@pytest.mark.parametrize('minspanx, x1', [[0, 10], [1, 10.5], [1, 11]])
@pytest.mark.parametrize('minspany, y1', [[0, 10], [1, 10.5], [1, 11]])
def test_rectangle_minspan(ax, spancoords, minspanx, x1, minspany, y1):
    onselect = mock.Mock(spec=noop, return_value=None)
    x0, y0 = (10, 10)
    if spancoords == 'pixels':
        minspanx, minspany = ax.transData.transform((x1, y1)) - ax.transData.transform((x0, y0))
    tool = widgets.RectangleSelector(ax, onselect, interactive=True, spancoords=spancoords, minspanx=minspanx, minspany=minspany)
    click_and_drag(tool, start=(x0, x1), end=(y0, y1))
    assert not tool._selection_completed
    onselect.assert_not_called()
    click_and_drag(tool, start=(20, 20), end=(30, 30))
    assert tool._selection_completed
    onselect.assert_called_once()
    onselect.reset_mock()
    click_and_drag(tool, start=(x0, y0), end=(x1, y1))
    assert not tool._selection_completed
    onselect.assert_called_once()
    (epress, erelease), kwargs = onselect.call_args
    assert epress.xdata == x0
    assert epress.ydata == y0
    assert erelease.xdata == x1
    assert erelease.ydata == y1
    assert kwargs == {}