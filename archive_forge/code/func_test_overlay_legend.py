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
def test_overlay_legend(self):
    overlay = Curve(range(10), label='A') * Curve(range(10), label='B')
    plot = bokeh_renderer.get_plot(overlay)
    legend_labels = [l.label['value'] for l in plot.state.legend[0].items]
    self.assertEqual(legend_labels, ['A', 'B'])