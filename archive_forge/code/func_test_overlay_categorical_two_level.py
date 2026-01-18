import numpy as np
import panel as pn
from bokeh.models import FactorRange, FixedTicker, HoverTool, Range1d, Span
from holoviews.core import DynamicMap, HoloMap, NdOverlay, Overlay
from holoviews.element import (
from holoviews.plotting.bokeh.util import property_to_dict
from holoviews.streams import Stream, Tap
from holoviews.util import Dynamic
from ...utils import LoggingComparisonTestCase
from .test_plot import TestBokehPlot, bokeh_renderer
def test_overlay_categorical_two_level(self):
    bars = Bars([('A', 'a', 1), ('B', 'b', 2), ('A', 'b', 3), ('B', 'a', 4)], kdims=['Upper', 'Lower'])
    plot = bokeh_renderer.get_plot(bars * HLine(2))
    x_range = plot.handles['x_range']
    assert isinstance(x_range, FactorRange)
    assert x_range.factors == [('A', 'a'), ('A', 'b'), ('B', 'a'), ('B', 'b')]
    assert isinstance(plot.state.renderers[-1], Span)