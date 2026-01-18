import unittest
import numpy as np
import holoviews as hv
from holoviews.element import (
from holoviews.plotting.bokeh.util import bokeh32, bokeh33
from .test_plot import TestBokehPlot, bokeh_renderer
def test_vspan_plot(self):
    vspan = VSpan(1.1, 1.5)
    plot = bokeh_renderer.get_plot(vspan)
    span = plot.handles['glyph']
    assert span.left == 1.1
    assert span.right == 1.5
    if bokeh33:
        assert isinstance(span.bottom, Node)
        assert isinstance(span.top, Node)
    else:
        assert span.bottom is None
        assert span.top is None
    assert span.visible