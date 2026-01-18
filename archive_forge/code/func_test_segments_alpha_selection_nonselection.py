import pandas as pd
from bokeh.models import FactorRange
from holoviews.core import NdOverlay
from holoviews.element import Segments
from .test_plot import TestBokehPlot, bokeh_renderer
def test_segments_alpha_selection_nonselection(self):
    opts = dict(alpha=0.8, selection_alpha=1.0, nonselection_alpha=0.2)
    segments = Segments([(i, i * 2, i * 3, i * 4, i * 5, chr(65 + i)) for i in range(10)], vdims=['a', 'b']).opts(**opts)
    plot = bokeh_renderer.get_plot(segments)
    glyph_renderer = plot.handles['glyph_renderer']
    self.assertEqual(glyph_renderer.glyph.line_alpha, 0.8)
    self.assertEqual(glyph_renderer.selection_glyph.line_alpha, 1)
    self.assertEqual(glyph_renderer.nonselection_glyph.line_alpha, 0.2)