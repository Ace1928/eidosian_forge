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
class TestResetCallback(CallbackTestCase):

    def test_reset_callback(self):
        resets = []

        def record(resetting):
            resets.append(resetting)
        curve = Curve([])
        stream = PlotReset(source=curve)
        stream.add_subscriber(record)
        plot = bokeh_server_renderer.get_plot(curve)
        plot.callbacks[0].on_msg({'reset': True})
        self.assertEqual(resets, [True])
        self.assertIs(stream.source, curve)