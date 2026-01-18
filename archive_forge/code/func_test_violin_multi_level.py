from unittest import SkipTest
import numpy as np
from bokeh.models import CategoricalColorMapper, LinearColorMapper
from holoviews.element import Violin
from holoviews.operation.stats import univariate_kde
from holoviews.plotting.bokeh.util import property_to_dict
from holoviews.util.transform import dim
from .test_plot import TestBokehPlot, bokeh_renderer
def test_violin_multi_level(self):
    box = Violin((['A', 'B'] * 15, [3, 10, 1] * 10, np.random.randn(30)), ['Group', 'Category'], 'Value')
    plot = bokeh_renderer.get_plot(box)
    x_range = plot.handles['x_range']
    self.assertEqual(x_range.factors, [('A', '1'), ('A', '3'), ('A', '10'), ('B', '1'), ('B', '3'), ('B', '10')])