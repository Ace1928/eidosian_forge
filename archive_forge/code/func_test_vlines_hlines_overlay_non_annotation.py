import unittest
import numpy as np
import holoviews as hv
from holoviews.element import (
from holoviews.plotting.bokeh.util import bokeh32, bokeh33
from .test_plot import TestBokehPlot, bokeh_renderer
def test_vlines_hlines_overlay_non_annotation(self):
    non_annotation = hv.Curve([], kdims=['time'])
    hspans = HSpans({'y0': [0, 3, 5.5], 'y1': [1, 4, 6.5], 'extra': [-1, -2, -3]}, vdims=['extra'])
    vspans = VSpans({'x0': [0, 3, 5.5], 'x1': [1, 4, 6.5], 'extra': [-1, -2, -3]}, vdims=['extra'])
    plot = bokeh_renderer.get_plot(non_annotation * hspans * vspans)
    assert plot.handles['xaxis'].axis_label == 'time'
    assert plot.handles['yaxis'].axis_label == 'y'