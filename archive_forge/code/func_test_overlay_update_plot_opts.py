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
def test_overlay_update_plot_opts(self):
    hmap = HoloMap({0: (Curve([]) * Curve([])).opts(title='A'), 1: (Curve([]) * Curve([])).opts(title='B')})
    plot = bokeh_renderer.get_plot(hmap)
    self.assertEqual(plot.state.title.text, 'A')
    plot.update((1,))
    self.assertEqual(plot.state.title.text, 'B')