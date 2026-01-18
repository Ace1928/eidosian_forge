import unittest
import numpy as np
import holoviews as hv
from holoviews.element import (
from holoviews.plotting.bokeh.util import bokeh32, bokeh33
from .test_plot import TestBokehPlot, bokeh_renderer
def test_text_plot_fontsize(self):
    text = Text(0, 0, 'Test', fontsize=18)
    plot = bokeh_renderer.get_plot(text)
    glyph = plot.handles['glyph']
    self.assertEqual(glyph.text_font_size, '18Pt')