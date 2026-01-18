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
def test_rectangle_selector_set_props_handle_props(ax):
    tool = widgets.RectangleSelector(ax, onselect=noop, interactive=True, props=dict(facecolor='b', alpha=0.2), handle_props=dict(alpha=0.5))
    click_and_drag(tool, start=(0, 10), end=(100, 120))
    artist = tool._selection_artist
    assert artist.get_facecolor() == mcolors.to_rgba('b', alpha=0.2)
    tool.set_props(facecolor='r', alpha=0.3)
    assert artist.get_facecolor() == mcolors.to_rgba('r', alpha=0.3)
    for artist in tool._handles_artists:
        assert artist.get_markeredgecolor() == 'black'
        assert artist.get_alpha() == 0.5
    tool.set_handle_props(markeredgecolor='r', alpha=0.3)
    for artist in tool._handles_artists:
        assert artist.get_markeredgecolor() == 'r'
        assert artist.get_alpha() == 0.3