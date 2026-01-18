import unittest
import numpy as np
import holoviews as hv
from holoviews.element import (
from holoviews.plotting.bokeh.util import bokeh32, bokeh33
from .test_plot import TestBokehPlot, bokeh_renderer
def test_vspan_invert_axes(self):
    vspan = VSpan(1.1, 1.5).opts(invert_axes=True)
    plot = bokeh_renderer.get_plot(vspan)
    span = plot.handles['glyph']
    if bokeh33:
        assert isinstance(span.left, Node)
        assert isinstance(span.right, Node)
    else:
        assert span.left is None
        assert span.right is None
    assert span.bottom == 1.1
    assert span.top == 1.5
    assert span.visible