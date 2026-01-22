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
class TestPointerCallbacks(CallbackTestCase):

    def test_pointer_x_datetime_out_of_bounds(self):
        points = Points([(dt.datetime(2017, 1, 1), 1), (dt.datetime(2017, 1, 3), 3)]).opts(padding=0)
        PointerX(source=points)
        plot = bokeh_server_renderer.get_plot(points)
        set_curdoc(plot.document)
        callback = plot.callbacks[0]
        self.assertIsInstance(callback, PointerXCallback)
        msg = callback._process_msg({'x': 1000})
        self.assertEqual(msg['x'], np.datetime64(dt.datetime(2017, 1, 1)))
        msg = callback._process_msg({'x': 10000000000000})
        self.assertEqual(msg['x'], np.datetime64(dt.datetime(2017, 1, 3)))

    def test_tap_datetime_out_of_bounds(self):
        points = Points([(dt.datetime(2017, 1, 1), 1), (dt.datetime(2017, 1, 3), 3)])
        SingleTap(source=points)
        plot = bokeh_server_renderer.get_plot(points)
        set_curdoc(plot.document)
        callback = plot.callbacks[0]
        self.assertIsInstance(callback, TapCallback)
        msg = callback._process_msg({'x': 1000, 'y': 2})
        self.assertEqual(msg, {})
        msg = callback._process_msg({'x': 10000000000000, 'y': 1})
        self.assertEqual(msg, {})