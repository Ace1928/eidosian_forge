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
def test_point_color_index_color_clash(self):
    points = Points([(0, 0, 0), (0, 1, 1), (0, 2, 2)], vdims='color').opts(color='color', color_index='color')
    with ParamLogStream() as log:
        bokeh_renderer.get_plot(points)
    log_msg = log.stream.read()
    warning = "The `color_index` parameter is deprecated in favor of color style mapping, e.g. `color=dim('color')` or `line_color=dim('color')`\nCannot declare style mapping for 'color' option and declare a color_index; ignoring the color_index.\n"
    self.assertEqual(log_msg, warning)