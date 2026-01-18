import datetime as dt
import numpy as np
from bokeh.models import CategoricalColorMapper, LinearColorMapper
from holoviews.core import NdOverlay
from holoviews.element import Spikes
from holoviews.plotting.bokeh.util import property_to_dict
from ..utils import ParamLogStream
from .test_plot import TestBokehPlot, bokeh_renderer
def test_spikes_datetime_kdim_hover(self):
    points = Spikes([(dt.datetime(2017, 1, 1), 1)], 'x', 'y').opts(tools=['hover'])
    plot = bokeh_renderer.get_plot(points)
    cds = plot.handles['cds']
    self.assertEqual(cds.data['x'].astype('datetime64'), np.array([1483228800000000000]))
    self.assertEqual(cds.data['y0'], np.array([0]))
    self.assertEqual(cds.data['y1'], np.array([1]))
    hover = plot.handles['hover']
    self.assertEqual(hover.tooltips, [('x', '@{x}{%F %T}'), ('y', '@{y}')])
    self.assertEqual(hover.formatters, {'@{x}': 'datetime'})