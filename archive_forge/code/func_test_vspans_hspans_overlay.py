import unittest
import numpy as np
import holoviews as hv
from holoviews.element import (
from holoviews.plotting.bokeh.util import bokeh32, bokeh33
from .test_plot import TestBokehPlot, bokeh_renderer
def test_vspans_hspans_overlay(self):
    hspans = HSpans({'y0': [0, 3, 5.5], 'y1': [1, 4, 6.5], 'extra': [-1, -2, -3]}, vdims=['extra'])
    vspans = VSpans({'x0': [0, 3, 5.5], 'x1': [1, 4, 6.5], 'extra': [-1, -2, -3]}, vdims=['extra'])
    plot = bokeh_renderer.get_plot(hspans * vspans)
    assert plot.handles['xaxis'].axis_label == 'x'
    assert plot.handles['yaxis'].axis_label == 'y'
    assert plot.handles['x_range'].start == 0
    assert plot.handles['x_range'].end == 6.5
    assert plot.handles['y_range'].start == 0
    assert plot.handles['y_range'].end == 6.5