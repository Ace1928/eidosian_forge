import unittest
import numpy as np
import holoviews as hv
from holoviews.element import (
from holoviews.plotting.bokeh.util import bokeh32, bokeh33
from .test_plot import TestBokehPlot, bokeh_renderer
class TestHVSpanPlot(TestBokehPlot):

    def test_hspan_invert_axes(self):
        hspan = HSpan(1.1, 1.5).opts(invert_axes=True)
        plot = bokeh_renderer.get_plot(hspan)
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

    def test_hspan_plot(self):
        hspan = HSpan(1.1, 1.5)
        plot = bokeh_renderer.get_plot(hspan)
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

    def test_hspan_empty(self):
        vline = HSpan(None)
        plot = bokeh_renderer.get_plot(vline)
        span = plot.handles['glyph']
        self.assertEqual(span.visible, False)

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

    def test_vspan_empty(self):
        vline = VSpan(None)
        plot = bokeh_renderer.get_plot(vline)
        span = plot.handles['glyph']
        self.assertEqual(span.visible, False)