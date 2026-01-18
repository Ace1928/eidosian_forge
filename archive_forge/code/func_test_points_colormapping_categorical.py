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
def test_points_colormapping_categorical(self):
    points = Points([(i, i * 2, i * 3, chr(65 + i)) for i in range(10)], vdims=['a', 'b']).opts(color_index='b')
    plot = bokeh_renderer.get_plot(points)
    plot.initialize_plot()
    cmapper = plot.handles['color_mapper']
    self.assertIsInstance(cmapper, CategoricalColorMapper)
    self.assertEqual(cmapper.factors, list(points['b']))