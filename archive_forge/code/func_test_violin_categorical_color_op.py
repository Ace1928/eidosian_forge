from unittest import SkipTest
import numpy as np
from bokeh.models import CategoricalColorMapper, LinearColorMapper
from holoviews.element import Violin
from holoviews.operation.stats import univariate_kde
from holoviews.plotting.bokeh.util import property_to_dict
from holoviews.util.transform import dim
from .test_plot import TestBokehPlot, bokeh_renderer
def test_violin_categorical_color_op(self):
    a = np.repeat(np.arange(5), 5)
    b = np.repeat(['A', 'B', 'C', 'D', 'E'], 5)
    violin = Violin((a, b, np.arange(25)), ['a', 'b'], 'd').opts(violin_color='b')
    plot = bokeh_renderer.get_plot(violin)
    source = plot.handles['patches_1_source']
    glyph = plot.handles['patches_1_glyph']
    cmapper = plot.handles['violin_color_color_mapper']
    self.assertEqual(source.data['violin_color'], b[::5])
    self.assertTrue(cmapper, CategoricalColorMapper)
    self.assertEqual(cmapper.factors, ['A', 'B', 'C', 'D', 'E'])
    self.assertEqual(property_to_dict(glyph.fill_color), {'field': 'violin_color', 'transform': cmapper})