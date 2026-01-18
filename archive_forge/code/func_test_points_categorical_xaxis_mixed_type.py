import datetime as dt
import numpy as np
import pandas as pd
from bokeh.models import CategoricalColorMapper, FactorRange, LinearColorMapper, Scatter
from holoviews.core import NdOverlay
from holoviews.core.options import Cycle
from holoviews.element import Points
from holoviews.plotting.bokeh.util import property_to_dict
from holoviews.streams import Stream
from ..utils import ParamLogStream
from .test_plot import TestBokehPlot, bokeh_renderer
def test_points_categorical_xaxis_mixed_type(self):
    points = Points(range(10))
    points2 = Points((['A', 'B', 'C', 1, 2.0], (1, 2, 3, 4, 5)))
    plot = bokeh_renderer.get_plot(points * points2)
    x_range = plot.handles['x_range']
    self.assertIsInstance(x_range, FactorRange)
    self.assertEqual(x_range.factors, list(map(str, range(10))) + ['A', 'B', 'C', '2.0'])