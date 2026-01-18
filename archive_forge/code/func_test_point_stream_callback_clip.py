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
def test_point_stream_callback_clip(self):
    dmap = DynamicMap(lambda x, y: Points([(x, y)]), kdims=[], streams=[PointerXY()])
    plot = bokeh_server_renderer.get_plot(dmap)
    bokeh_server_renderer(plot)
    set_curdoc(plot.document)
    plot.callbacks[0].on_msg({'x': -0.3, 'y': 1.2})
    data = plot.handles['source'].data
    self.assertEqual(data['x'], np.array([0]))
    self.assertEqual(data['y'], np.array([1]))