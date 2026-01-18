import numpy as np
import pytest
from bokeh.models import ColumnDataSource, RangeTool
from holoviews.core.spaces import DynamicMap
from holoviews.element import Curve, Image, Path, Points, Polygons, Scatter, Table
from holoviews.plotting.links import DataLink, Link, RangeToolLink
from .test_plot import TestBokehPlot, bokeh_renderer
def test_range_tool_link_callback_boundsy_arg(self):
    array = np.random.rand(100, 2)
    src = Curve(array)
    target = Scatter(array)
    y_start = 0.8
    y_end = 0.9
    RangeToolLink(src, target, axes=['x', 'y'], boundsy=(y_start, y_end))
    layout = target + src
    plot = bokeh_renderer.get_plot(layout)
    tgt_plot = plot.subplots[0, 0].subplots['main']
    self.assertEqual(tgt_plot.handles['y_range'].start, y_start)
    self.assertEqual(tgt_plot.handles['y_range'].end, y_end)
    self.assertEqual(tgt_plot.handles['y_range'].reset_start, y_start)
    self.assertEqual(tgt_plot.handles['y_range'].reset_end, y_end)