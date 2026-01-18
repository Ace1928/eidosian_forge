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
def test_span_selector_direction(ax):
    tool = widgets.SpanSelector(ax, onselect=noop, direction='horizontal', interactive=True)
    assert tool.direction == 'horizontal'
    assert tool._edge_handles.direction == 'horizontal'
    with pytest.raises(ValueError):
        tool = widgets.SpanSelector(ax, onselect=noop, direction='invalid_direction')
    tool.direction = 'vertical'
    assert tool.direction == 'vertical'
    assert tool._edge_handles.direction == 'vertical'
    with pytest.raises(ValueError):
        tool.direction = 'invalid_string'