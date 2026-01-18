from unittest import SkipTest
import numpy as np
from bokeh.models import CategoricalColorMapper, LinearColorMapper
from holoviews.element import Violin
from holoviews.operation.stats import univariate_kde
from holoviews.plotting.bokeh.util import property_to_dict
from holoviews.util.transform import dim
from .test_plot import TestBokehPlot, bokeh_renderer
def test_violin_line_width_op(self):
    a = np.repeat(np.arange(5), 5)
    b = np.repeat(np.arange(5), 5)
    violin = Violin((a, b, np.arange(25)), ['a', 'b'], 'd').opts(violin_line_width='b')
    plot = bokeh_renderer.get_plot(violin)
    source = plot.handles['multi_line_1_source']
    glyph = plot.handles['multi_line_1_glyph']
    self.assertEqual(source.data['outline_line_width'], np.arange(5))
    self.assertEqual(property_to_dict(glyph.line_width), {'field': 'outline_line_width'})