from unittest import SkipTest
import numpy as np
from bokeh.models import CategoricalColorMapper, LinearColorMapper
from holoviews.element import Violin
from holoviews.operation.stats import univariate_kde
from holoviews.plotting.bokeh.util import property_to_dict
from holoviews.util.transform import dim
from .test_plot import TestBokehPlot, bokeh_renderer
def test_violin_empty(self):
    violin = Violin([])
    plot = bokeh_renderer.get_plot(violin)
    patch_source = plot.handles['patches_1_source']
    self.assertEqual(patch_source.data['xs'], [[]])
    self.assertEqual(patch_source.data['ys'], [np.array([])])