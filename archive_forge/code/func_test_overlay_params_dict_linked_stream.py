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
def test_overlay_params_dict_linked_stream(self):
    tap = Tap()

    def test(x):
        return Curve([1, 2, 3]) * VLine(x or 0)
    dmap = DynamicMap(test, streams={'x': tap.param.x})
    plot = bokeh_renderer.get_plot(dmap)
    tap.event(x=1)
    _, vline_plot = plot.subplots.values()
    assert vline_plot.handles['glyph'].location == 1