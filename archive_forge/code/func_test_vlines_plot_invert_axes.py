import unittest
import numpy as np
import holoviews as hv
from holoviews.element import (
from holoviews.plotting.bokeh.util import bokeh32, bokeh33
from .test_plot import TestBokehPlot, bokeh_renderer
def test_vlines_plot_invert_axes(self):
    vlines = VLines({'x': [0, 1, 2, 5.5], 'extra': [-1, -2, -3, -44]}, vdims=['extra']).opts(invert_axes=True)
    plot = bokeh_renderer.get_plot(vlines)
    assert isinstance(plot.handles['glyph'], BkHSpan)
    assert plot.handles['xaxis'].axis_label == 'y'
    assert plot.handles['yaxis'].axis_label == 'x'
    assert plot.handles['x_range'].start == 0
    assert plot.handles['x_range'].end == 1
    assert plot.handles['y_range'].start == 0
    assert plot.handles['y_range'].end == 5.5
    source = plot.handles['source']
    assert list(source.data) == ['x']
    assert (source.data['x'] == [0, 1, 2, 5.5]).all()