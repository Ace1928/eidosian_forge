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
def test_poly_edit_shared_callback(self):
    polys = Polygons([[(0, 0), (2, 2), (4, 0)]])
    polys2 = Polygons([[(0, 0), (2, 2), (4, 0)]])
    poly_edit = PolyEdit(source=polys, shared=True)
    poly_edit2 = PolyEdit(source=polys2, shared=True)
    plot = bokeh_server_renderer.get_plot(polys * polys2)
    edit_tools = [t for t in plot.state.tools if isinstance(t, PolyEditTool)]
    self.assertEqual(len(edit_tools), 1)
    plot1, plot2 = plot.subplots.values()
    self.assertIsInstance(plot1.callbacks[0], PolyEditCallback)
    callback = plot1.callbacks[0]
    data = {'x': [[1, 2, 3], [3, 4, 5]], 'y': [[1, 2, 3], [3, 4, 5]]}
    callback.on_msg({'data': data})
    self.assertIsInstance(plot2.callbacks[0], PolyEditCallback)
    callback = plot2.callbacks[0]
    data = {'x': [[1, 2, 3], [3, 4, 5]], 'y': [[1, 2, 3], [3, 4, 5]]}
    callback.on_msg({'data': data})
    element = Polygons([[(1, 1), (2, 2), (3, 3)], [(3, 3), (4, 4), (5, 5)]])
    self.assertEqual(poly_edit.element, element)
    self.assertEqual(poly_edit2.element, element)