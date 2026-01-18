from unittest import SkipTest
import numpy as np
from bokeh.models import CategoricalColorMapper, LinearColorMapper
from holoviews.element import Violin
from holoviews.operation.stats import univariate_kde
from holoviews.plotting.bokeh.util import property_to_dict
from holoviews.util.transform import dim
from .test_plot import TestBokehPlot, bokeh_renderer
def test_violin_inner_stick(self):
    values = np.random.rand(100)
    violin = Violin(values).opts(inner='stick')
    kde = univariate_kde(violin, cut=5)
    xs = kde.dimension_values(0)
    plot = bokeh_renderer.get_plot(violin)
    self.assertIn('segment_1_glyph_renderer', plot.handles)
    segments = np.array([xs[np.argmin(np.abs(xs - v))] for v in values])
    self.assertEqual(plot.handles['segment_1_source'].data['x'], segments)