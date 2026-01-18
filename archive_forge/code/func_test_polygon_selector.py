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
@pytest.mark.parametrize('draw_bounding_box', [False, True])
def test_polygon_selector(draw_bounding_box):
    check_selector = functools.partial(check_polygon_selector, draw_bounding_box=draw_bounding_box)
    expected_result = [(50, 50), (150, 50), (50, 150)]
    event_sequence = [*polygon_place_vertex(50, 50), *polygon_place_vertex(150, 50), *polygon_place_vertex(50, 150), *polygon_place_vertex(50, 50)]
    check_selector(event_sequence, expected_result, 1)
    expected_result = [(75, 50), (150, 50), (50, 150)]
    event_sequence = [*polygon_place_vertex(50, 50), *polygon_place_vertex(150, 50), ('on_key_press', dict(key='control')), ('onmove', dict(xdata=50, ydata=50)), ('press', dict(xdata=50, ydata=50)), ('onmove', dict(xdata=75, ydata=50)), ('release', dict(xdata=75, ydata=50)), ('on_key_release', dict(key='control')), *polygon_place_vertex(50, 150), *polygon_place_vertex(75, 50)]
    check_selector(event_sequence, expected_result, 1)
    expected_result = [(50, 75), (150, 75), (50, 150)]
    event_sequence = [*polygon_place_vertex(50, 50), *polygon_place_vertex(150, 50), ('on_key_press', dict(key='shift')), ('onmove', dict(xdata=100, ydata=100)), ('press', dict(xdata=100, ydata=100)), ('onmove', dict(xdata=100, ydata=125)), ('release', dict(xdata=100, ydata=125)), ('on_key_release', dict(key='shift')), *polygon_place_vertex(50, 150), *polygon_place_vertex(50, 75)]
    check_selector(event_sequence, expected_result, 1)
    expected_result = [(75, 50), (150, 50), (50, 150)]
    event_sequence = [*polygon_place_vertex(50, 50), *polygon_place_vertex(150, 50), *polygon_place_vertex(50, 150), *polygon_place_vertex(50, 50), ('onmove', dict(xdata=50, ydata=50)), ('press', dict(xdata=50, ydata=50)), ('onmove', dict(xdata=75, ydata=50)), ('release', dict(xdata=75, ydata=50))]
    check_selector(event_sequence, expected_result, 2)
    expected_result = [(75, 75), (175, 75), (75, 175)]
    event_sequence = [*polygon_place_vertex(50, 50), *polygon_place_vertex(150, 50), *polygon_place_vertex(50, 150), *polygon_place_vertex(50, 50), ('on_key_press', dict(key='shift')), ('onmove', dict(xdata=100, ydata=100)), ('press', dict(xdata=100, ydata=100)), ('onmove', dict(xdata=125, ydata=125)), ('release', dict(xdata=125, ydata=125)), ('on_key_release', dict(key='shift'))]
    check_selector(event_sequence, expected_result, 2)
    expected_result = [(50, 50), (150, 50), (50, 150)]
    event_sequence = [('on_key_press', dict(key='control')), ('onmove', dict(xdata=100, ydata=100)), ('press', dict(xdata=100, ydata=100)), ('onmove', dict(xdata=125, ydata=125)), ('release', dict(xdata=125, ydata=125)), ('on_key_release', dict(key='control')), ('on_key_press', dict(key='shift')), ('onmove', dict(xdata=100, ydata=100)), ('press', dict(xdata=100, ydata=100)), ('onmove', dict(xdata=125, ydata=125)), ('release', dict(xdata=125, ydata=125)), ('on_key_release', dict(key='shift')), *polygon_place_vertex(50, 50), *polygon_place_vertex(150, 50), *polygon_place_vertex(50, 150), *polygon_place_vertex(50, 50)]
    check_selector(event_sequence, expected_result, 1)
    expected_result = [(50, 50), (150, 50), (50, 150)]
    event_sequence = [*polygon_place_vertex(50, 50), *polygon_place_vertex(250, 50), ('on_key_press', dict(key='escape')), ('on_key_release', dict(key='escape')), *polygon_place_vertex(50, 50), *polygon_place_vertex(150, 50), *polygon_place_vertex(50, 150), *polygon_place_vertex(50, 50)]
    check_selector(event_sequence, expected_result, 1)