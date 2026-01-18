import numpy as np
import pytest
from bokeh.models import ColumnDataSource, RangeTool
from holoviews.core.spaces import DynamicMap
from holoviews.element import Curve, Image, Path, Points, Polygons, Scatter, Table
from holoviews.plotting.links import DataLink, Link, RangeToolLink
from .test_plot import TestBokehPlot, bokeh_renderer
def test_range_tool_link_callback_single_axis_overlay_target(self):
    array = np.random.rand(100, 2)
    src = Curve(array)
    target = Scatter(array, label='a') * Scatter(array, label='b')
    RangeToolLink(src, target)
    layout = target + src
    plot = bokeh_renderer.get_plot(layout)
    tgt_plot = plot.subplots[0, 0].subplots['main']
    src_plot = plot.subplots[0, 1].subplots['main']
    range_tool = src_plot.state.select_one({'type': RangeTool})
    self.assertEqual(range_tool.x_range, tgt_plot.handles['x_range'])
    self.assertIs(range_tool.y_range, None)