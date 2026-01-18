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
def test_point_angle_op(self):
    points = Points([(0, 0, 0), (0, 1, 45), (0, 2, 90)], vdims='angle').opts(angle='angle')
    plot = bokeh_renderer.get_plot(points)
    cds = plot.handles['cds']
    glyph = plot.handles['glyph']
    self.assertEqual(cds.data['angle'], np.array([0, 0.785398, 1.570796]))
    self.assertEqual(property_to_dict(glyph.angle), {'field': 'angle'})