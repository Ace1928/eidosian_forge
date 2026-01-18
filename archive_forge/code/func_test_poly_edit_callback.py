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
@pytest.mark.flaky(reruns=3)
def test_poly_edit_callback(self):
    polys = Polygons([[(0, 0), (2, 2), (4, 0)]])
    poly_edit = PolyEdit(source=polys)
    plot = bokeh_server_renderer.get_plot(polys)
    self.assertIsInstance(plot.callbacks[0], PolyEditCallback)
    callback = plot.callbacks[0]
    data = {'x': [[1, 2, 3], [3, 4, 5]], 'y': [[1, 2, 3], [3, 4, 5]]}
    callback.on_msg({'data': data})
    element = Polygons([[(1, 1), (2, 2), (3, 3)], [(3, 3), (4, 4), (5, 5)]])
    self.assertEqual(poly_edit.element, element)