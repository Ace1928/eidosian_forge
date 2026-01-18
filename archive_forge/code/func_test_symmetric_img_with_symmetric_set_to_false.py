import os
import tempfile
from unittest import SkipTest
from collections import OrderedDict
import numpy as np
from holoviews import Store
from holoviews.element import RGB, Image
from holoviews.element.comparison import ComparisonTestCase
def test_symmetric_img_with_symmetric_set_to_false(self):
    plot = self.da_img.hvplot.image(symmetric=False)
    plot_opts = Store.lookup_options('bokeh', plot, 'plot')
    self.assertEqual(plot_opts.kwargs.get('symmetric'), False)
    style_opts = Store.lookup_options('bokeh', plot, 'style')
    self.assertEqual(style_opts.kwargs['cmap'], 'kbc_r')