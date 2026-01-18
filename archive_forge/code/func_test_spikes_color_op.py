import datetime as dt
import numpy as np
from bokeh.models import CategoricalColorMapper, LinearColorMapper
from holoviews.core import NdOverlay
from holoviews.element import Spikes
from holoviews.plotting.bokeh.util import property_to_dict
from ..utils import ParamLogStream
from .test_plot import TestBokehPlot, bokeh_renderer
def test_spikes_color_op(self):
    spikes = Spikes([(0, 0, '#000'), (0, 1, '#F00'), (0, 2, '#0F0')], vdims=['y', 'color']).opts(color='color')
    plot = bokeh_renderer.get_plot(spikes)
    cds = plot.handles['cds']
    glyph = plot.handles['glyph']
    self.assertEqual(cds.data['color'], np.array(['#000', '#F00', '#0F0']))
    self.assertEqual(property_to_dict(glyph.line_color), {'field': 'color'})