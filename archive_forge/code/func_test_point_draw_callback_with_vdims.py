import datetime as dt
from collections import deque, namedtuple
from unittest import SkipTest
import numpy as np
import pandas as pd
import pytest
import pyviz_comms as comms
from bokeh.events import Tap
from bokeh.io.doc import set_curdoc
from bokeh.models import ColumnDataSource, Plot, PolyEditTool, Range1d, Selection
from holoviews.core import DynamicMap
from holoviews.core.options import Store
from holoviews.element import Box, Curve, Points, Polygons, Rectangles, Table
from holoviews.element.comparison import ComparisonTestCase
from holoviews.plotting.bokeh.callbacks import (
from holoviews.plotting.bokeh.renderer import BokehRenderer
from holoviews.streams import (
def test_point_draw_callback_with_vdims(self):
    points = Points([(0, 1, 'A')], vdims=['A'])
    point_draw = PointDraw(source=points)
    plot = bokeh_server_renderer.get_plot(points)
    self.assertIsInstance(plot.callbacks[0], PointDrawCallback)
    callback = plot.callbacks[0]
    data = {'x': [1, 2, 3], 'y': [1, 2, 3], 'A': [None, None, 1]}
    callback.on_msg({'data': data})
    processed = dict(data, A=[np.nan, np.nan, 1])
    self.assertEqual(point_draw.element, Points(processed, vdims=['A']))