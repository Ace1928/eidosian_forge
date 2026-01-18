import datetime as dt
import numpy as np
from bokeh.models import CategoricalColorMapper, LinearColorMapper
from holoviews.core import NdOverlay
from holoviews.element import Spikes
from holoviews.plotting.bokeh.util import property_to_dict
from ..utils import ParamLogStream
from .test_plot import TestBokehPlot, bokeh_renderer
def test_spikes_datetime_vdim_hover(self):
    points = Spikes([(0, 1, dt.datetime(2017, 1, 1))], vdims=['value', 'date']).opts(tools=['hover'])
    plot = bokeh_renderer.get_plot(points)
    cds = plot.handles['cds']
    self.assertEqual(cds.data['date'].astype('datetime64'), np.array([1483228800000000000]))
    hover = plot.handles['hover']
    self.assertEqual(hover.tooltips, [('x', '@{x}'), ('value', '@{value}'), ('date', '@{date}{%F %T}')])
    self.assertEqual(hover.formatters, {'@{date}': 'datetime'})