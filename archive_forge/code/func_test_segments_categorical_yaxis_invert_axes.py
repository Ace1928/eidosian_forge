import pandas as pd
from bokeh.models import FactorRange
from holoviews.core import NdOverlay
from holoviews.element import Segments
from .test_plot import TestBokehPlot, bokeh_renderer
def test_segments_categorical_yaxis_invert_axes(self):
    segments = Segments(([1, 2, 3], ['A', 'B', 'C'], [4, 5, 6], ['A', 'B', 'C']))
    plot = bokeh_renderer.get_plot(segments)
    y_range = plot.handles['y_range']
    self.assertIsInstance(y_range, FactorRange)
    self.assertEqual(y_range.factors, ['A', 'B', 'C'])