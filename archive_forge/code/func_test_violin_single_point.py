from unittest import SkipTest
import numpy as np
from bokeh.models import CategoricalColorMapper, LinearColorMapper
from holoviews.element import Violin
from holoviews.operation.stats import univariate_kde
from holoviews.plotting.bokeh.util import property_to_dict
from holoviews.util.transform import dim
from .test_plot import TestBokehPlot, bokeh_renderer
def test_violin_single_point(self):
    data = {'x': [1], 'y': [1]}
    violin = Violin(data=data, kdims='x', vdims='y').opts(inner='box')
    plot = bokeh_renderer.get_plot(violin)
    self.assertEqual(plot.handles['x_range'].factors, ['1'])