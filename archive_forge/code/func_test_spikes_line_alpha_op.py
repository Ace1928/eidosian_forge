import datetime as dt
import numpy as np
from bokeh.models import CategoricalColorMapper, LinearColorMapper
from holoviews.core import NdOverlay
from holoviews.element import Spikes
from holoviews.plotting.bokeh.util import property_to_dict
from ..utils import ParamLogStream
from .test_plot import TestBokehPlot, bokeh_renderer
def test_spikes_line_alpha_op(self):
    spikes = Spikes([(0, 0, 0), (0, 1, 0.2), (0, 2, 0.7)], vdims=['y', 'alpha']).opts(line_alpha='alpha')
    plot = bokeh_renderer.get_plot(spikes)
    cds = plot.handles['cds']
    glyph = plot.handles['glyph']
    self.assertEqual(cds.data['line_alpha'], np.array([0, 0.2, 0.7]))
    self.assertEqual(property_to_dict(glyph.line_alpha), {'field': 'line_alpha'})