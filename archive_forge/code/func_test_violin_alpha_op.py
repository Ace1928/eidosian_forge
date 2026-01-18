from unittest import SkipTest
import numpy as np
from bokeh.models import CategoricalColorMapper, LinearColorMapper
from holoviews.element import Violin
from holoviews.operation.stats import univariate_kde
from holoviews.plotting.bokeh.util import property_to_dict
from holoviews.util.transform import dim
from .test_plot import TestBokehPlot, bokeh_renderer
def test_violin_alpha_op(self):
    a = np.repeat(np.arange(5), 5)
    b = np.repeat(np.arange(5) / 10.0, 5)
    violin = Violin((a, b, np.arange(25)), ['a', 'b'], 'd').opts(violin_alpha='b')
    plot = bokeh_renderer.get_plot(violin)
    source = plot.handles['patches_1_source']
    glyph = plot.handles['patches_1_glyph']
    self.assertEqual(source.data['violin_alpha'], np.arange(5) / 10.0)
    self.assertEqual(property_to_dict(glyph.fill_alpha), {'field': 'violin_alpha'})