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
def test_points_selected(self):
    points = Points([(0, 0), (1, 1), (2, 2)]).opts(selected=[0, 2])
    plot = bokeh_renderer.get_plot(points)
    cds = plot.handles['cds']
    self.assertEqual(cds.selected.indices, [0, 2])