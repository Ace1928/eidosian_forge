import datetime as dt
import numpy as np
from bokeh.models import CategoricalColorMapper, LinearColorMapper
from holoviews.core import NdOverlay
from holoviews.element import Spikes
from holoviews.plotting.bokeh.util import property_to_dict
from ..utils import ParamLogStream
from .test_plot import TestBokehPlot, bokeh_renderer
def test_spikes_categorical_color_op(self):
    spikes = Spikes([(0, 0, 'A'), (0, 1, 'B'), (0, 2, 'C')], vdims=['y', 'color']).opts(color='color')
    plot = bokeh_renderer.get_plot(spikes)
    cds = plot.handles['cds']
    glyph = plot.handles['glyph']
    cmapper = plot.handles['color_color_mapper']
    self.assertTrue(cmapper, CategoricalColorMapper)
    self.assertEqual(cmapper.factors, ['A', 'B', 'C'])
    self.assertEqual(cds.data['color'], np.array(['A', 'B', 'C']))
    self.assertEqual(property_to_dict(glyph.line_color), {'field': 'color', 'transform': cmapper})