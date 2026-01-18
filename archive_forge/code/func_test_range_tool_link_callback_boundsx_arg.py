import numpy as np
import pytest
from bokeh.models import ColumnDataSource, RangeTool
from holoviews.core.spaces import DynamicMap
from holoviews.element import Curve, Image, Path, Points, Polygons, Scatter, Table
from holoviews.plotting.links import DataLink, Link, RangeToolLink
from .test_plot import TestBokehPlot, bokeh_renderer
def test_range_tool_link_callback_boundsx_arg(self):
    array = np.random.rand(100, 2)
    src = Curve(array)
    target = Scatter(array)
    x_start = 0.2
    x_end = 0.3
    RangeToolLink(src, target, axes=['x', 'y'], boundsx=(x_start, x_end))
    layout = target + src
    plot = bokeh_renderer.get_plot(layout)
    tgt_plot = plot.subplots[0, 0].subplots['main']
    self.assertEqual(tgt_plot.handles['x_range'].start, x_start)
    self.assertEqual(tgt_plot.handles['x_range'].end, x_end)
    self.assertEqual(tgt_plot.handles['x_range'].reset_start, x_start)
    self.assertEqual(tgt_plot.handles['x_range'].reset_end, x_end)