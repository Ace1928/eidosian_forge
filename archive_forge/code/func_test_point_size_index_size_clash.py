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
def test_point_size_index_size_clash(self):
    points = Points([(0, 0, 0), (0, 1, 1), (0, 2, 2)], vdims='size').opts(size='size', size_index='size')
    with ParamLogStream() as log:
        bokeh_renderer.get_plot(points)
    log_msg = log.stream.read()
    warning = "The `size_index` parameter is deprecated in favor of size style mapping, e.g. `size=dim('size')**2`.\nCannot declare style mapping for 'size' option and declare a size_index; ignoring the size_index.\n"
    self.assertEqual(log_msg, warning)