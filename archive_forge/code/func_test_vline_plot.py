import unittest
import numpy as np
import holoviews as hv
from holoviews.element import (
from holoviews.plotting.bokeh.util import bokeh32, bokeh33
from .test_plot import TestBokehPlot, bokeh_renderer
def test_vline_plot(self):
    vline = VLine(1.1)
    plot = bokeh_renderer.get_plot(vline)
    span = plot.handles['glyph']
    self.assertEqual(span.dimension, 'height')
    self.assertEqual(span.location, 1.1)