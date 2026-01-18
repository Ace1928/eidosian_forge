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
@pytest.mark.parametrize('ignore_event_outside', [True, False])
def test_span_selector_ignore_outside(ax, ignore_event_outside):
    onselect = mock.Mock(spec=noop, return_value=None)
    onmove = mock.Mock(spec=noop, return_value=None)
    tool = widgets.SpanSelector(ax, onselect, 'horizontal', onmove_callback=onmove, ignore_event_outside=ignore_event_outside)
    click_and_drag(tool, start=(100, 100), end=(125, 125))
    onselect.assert_called_once()
    onmove.assert_called_once()
    assert tool.extents == (100, 125)
    onselect.reset_mock()
    onmove.reset_mock()
    click_and_drag(tool, start=(150, 150), end=(160, 160))
    if ignore_event_outside:
        onselect.assert_not_called()
        onmove.assert_not_called()
        assert tool.extents == (100, 125)
    else:
        onselect.assert_called_once()
        onmove.assert_called_once()
        assert tool.extents == (150, 160)