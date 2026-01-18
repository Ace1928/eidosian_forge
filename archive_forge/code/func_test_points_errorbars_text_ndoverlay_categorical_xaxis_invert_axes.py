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
def test_points_errorbars_text_ndoverlay_categorical_xaxis_invert_axes(self):
    overlay = NdOverlay({i: Points(([chr(65 + i)] * 10, np.random.randn(10))) for i in range(5)})
    error = ErrorBars([(el['x'][0], np.mean(el['y']), np.std(el['y'])) for el in overlay]).opts(invert_axes=True)
    text = Text('C', 0, 'Test')
    plot = bokeh_renderer.get_plot(overlay * error * text)
    x_range = plot.handles['x_range']
    y_range = plot.handles['y_range']
    self.assertIsInstance(x_range, Range1d)
    self.assertIsInstance(y_range, FactorRange)
    self.assertEqual(y_range.factors, ['A', 'B', 'C', 'D', 'E'])