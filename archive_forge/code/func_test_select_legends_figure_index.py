import pytest
import holoviews as hv
from holoviews.core import Store
from holoviews.element.comparison import ComparisonTestCase
from holoviews.plotting.bokeh.styles import expand_batched_style
from holoviews.plotting.bokeh.util import (
@pytest.mark.parametrize('figure_index,expected', [(0, [True, False]), (1, [False, True]), ([0], [True, False]), ([1], [False, True]), ([0, 1], [True, True]), (True, [True, True]), (False, [False, False]), (None, [True, False])], ids=['int0', 'int1', 'list0', 'list1', 'list01', 'True', 'False', 'None'])
def test_select_legends_figure_index(figure_index, expected):
    overlays = [hv.Curve([0, 0]) * hv.Curve([1, 1]), hv.Curve([2, 2]) * hv.Curve([3, 3])]
    layout = hv.Layout(overlays)
    select_legends(layout, figure_index)
    output = [ol.opts['show_legend'] for ol in overlays]
    assert expected == output