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
def test_selector_clear(ax, selector):
    kwargs = dict(ax=ax, onselect=noop, interactive=True)
    if selector == 'span':
        Selector = widgets.SpanSelector
        kwargs['direction'] = 'horizontal'
    else:
        Selector = widgets.RectangleSelector
    tool = Selector(**kwargs)
    click_and_drag(tool, start=(10, 10), end=(100, 120))
    click_and_drag(tool, start=(130, 130), end=(130, 130))
    assert not tool._selection_completed
    kwargs['ignore_event_outside'] = True
    tool = Selector(**kwargs)
    assert tool.ignore_event_outside
    click_and_drag(tool, start=(10, 10), end=(100, 120))
    click_and_drag(tool, start=(130, 130), end=(130, 130))
    assert tool._selection_completed
    do_event(tool, 'on_key_press', key='escape')
    assert not tool._selection_completed