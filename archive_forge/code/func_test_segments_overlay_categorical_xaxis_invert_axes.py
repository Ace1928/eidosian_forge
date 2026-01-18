import pandas as pd
from bokeh.models import FactorRange
from holoviews.core import NdOverlay
from holoviews.element import Segments
from .test_plot import TestBokehPlot, bokeh_renderer
def test_segments_overlay_categorical_xaxis_invert_axes(self):
    segments = Segments(([1, 2, 3], ['A', 'B', 'C'], [4, 5, 6], ['A', 'B', 'C'])).opts(invert_axes=True)
    segments2 = Segments(([1, 2, 3], ['B', 'C', 'D'], [4, 5, 6], ['B', 'C', 'D']))
    plot = bokeh_renderer.get_plot(segments * segments2)
    x_range = plot.handles['x_range']
    self.assertIsInstance(x_range, FactorRange)
    self.assertEqual(x_range.factors, ['A', 'B', 'C', 'D'])