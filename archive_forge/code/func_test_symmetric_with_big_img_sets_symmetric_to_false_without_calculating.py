import os
import tempfile
from unittest import SkipTest
from collections import OrderedDict
import numpy as np
from holoviews import Store
from holoviews.element import RGB, Image
from holoviews.element.comparison import ComparisonTestCase
def test_symmetric_with_big_img_sets_symmetric_to_false_without_calculating(self):
    plot = self.big_img.hvplot.image()
    plot_opts = Store.lookup_options('bokeh', plot, 'plot')
    self.assertEqual(plot_opts.kwargs.get('symmetric'), False)
    style_opts = Store.lookup_options('bokeh', plot, 'style')
    self.assertEqual(style_opts.kwargs['cmap'], 'kbc_r')