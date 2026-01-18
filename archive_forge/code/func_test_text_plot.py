import unittest
import numpy as np
import holoviews as hv
from holoviews.element import (
from holoviews.plotting.bokeh.util import bokeh32, bokeh33
from .test_plot import TestBokehPlot, bokeh_renderer
def test_text_plot(self):
    text = Text(0, 0, 'Test')
    plot = bokeh_renderer.get_plot(text)
    source = plot.handles['source']
    self.assertEqual(source.data, {'x': [0], 'y': [0], 'text': ['Test']})