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
def test_rangexy_framewise_not_reset_if_triggering(self):
    stream = RangeXY(x_range=(0, 2), y_range=(0, 1))
    curve = DynamicMap(lambda z, x_range, y_range: Curve([1, 2, z]), kdims=['z'], streams=[stream]).redim.range(z=(0, 3))
    bokeh_server_renderer.get_plot(curve.opts(framewise=True))
    stream.event(x_range=(0, 3))
    self.assertEqual(stream.x_range, (0, 3))