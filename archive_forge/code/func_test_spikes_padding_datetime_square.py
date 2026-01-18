import datetime as dt
import numpy as np
from bokeh.models import CategoricalColorMapper, LinearColorMapper
from holoviews.core import NdOverlay
from holoviews.element import Spikes
from holoviews.plotting.bokeh.util import property_to_dict
from ..utils import ParamLogStream
from .test_plot import TestBokehPlot, bokeh_renderer
def test_spikes_padding_datetime_square(self):
    spikes = Spikes([np.datetime64('2016-04-0%d' % i) for i in range(1, 4)]).opts(padding=0.1)
    plot = bokeh_renderer.get_plot(spikes)
    x_range = plot.handles['x_range']
    self.assertEqual(x_range.start, np.datetime64('2016-03-31T19:12:00.000000000'))
    self.assertEqual(x_range.end, np.datetime64('2016-04-03T04:48:00.000000000'))