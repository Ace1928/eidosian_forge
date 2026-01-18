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
@pytest.mark.parametrize('selector', ['span', 'rectangle'])
def test_selector_clear_method(ax, selector):
    if selector == 'span':
        tool = widgets.SpanSelector(ax, onselect=noop, direction='horizontal', interactive=True, ignore_event_outside=True)
    else:
        tool = widgets.RectangleSelector(ax, onselect=noop, interactive=True)
    click_and_drag(tool, start=(10, 10), end=(100, 120))
    assert tool._selection_completed
    assert tool.get_visible()
    if selector == 'span':
        assert tool.extents == (10, 100)
    tool.clear()
    assert not tool._selection_completed
    assert not tool.get_visible()
    click_and_drag(tool, start=(10, 10), end=(50, 120))
    assert tool._selection_completed
    assert tool.get_visible()
    if selector == 'span':
        assert tool.extents == (10, 50)