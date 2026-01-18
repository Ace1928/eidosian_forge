import os
import tempfile
from unittest import SkipTest
from collections import OrderedDict
import numpy as np
from holoviews import Store
from holoviews.element import RGB, Image
from holoviews.element.comparison import ComparisonTestCase
def test_symmetric_with_big_img_and_check_symmetric_max_calculates_symmetric(self):
    plot = self.big_img.hvplot.image(check_symmetric_max=int(10000000.0))
    plot_opts = Store.lookup_options('bokeh', plot, 'plot')
    self.assertEqual(plot_opts.kwargs.get('symmetric'), True)
    style_opts = Store.lookup_options('bokeh', plot, 'style')
    self.assertEqual(style_opts.kwargs['cmap'], 'coolwarm')