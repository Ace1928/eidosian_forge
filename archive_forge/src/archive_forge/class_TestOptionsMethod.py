import os
import pickle
import numpy as np
import pytest
from holoviews import (
from holoviews.core.options import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.plotting import mpl # noqa
from holoviews.plotting import bokeh # noqa
from holoviews.plotting import plotly # noqa
class TestOptionsMethod(ComparisonTestCase):

    def setUp(self):
        self.backend = 'matplotlib'
        Store.set_current_backend(self.backend)
        self.store_copy = OptionTree(sorted(Store.options().items()), groups=Options._option_groups)
        super().setUp()

    def tearDown(self):
        Store.options(val=self.store_copy)
        Store._custom_options = {k: {} for k in Store._custom_options.keys()}
        super().tearDown()

    def lookup_options(self, obj, group):
        return Store.lookup_options(self.backend, obj, group)

    def test_plot_options_keywords(self):
        im = Image(np.random.rand(10, 10))
        styled_im = im.options(interpolation='nearest', cmap='jet')
        self.assertEqual(self.lookup_options(im, 'plot').options, {})
        self.assertEqual(self.lookup_options(styled_im, 'style').options, dict(cmap='jet', interpolation='nearest'))

    def test_plot_options_one_object(self):
        im = Image(np.random.rand(10, 10))
        imopts = opts.Image(interpolation='nearest', cmap='jet')
        styled_im = im.options(imopts)
        self.assertEqual(self.lookup_options(im, 'plot').options, {})
        self.assertEqual(self.lookup_options(styled_im, 'style').options, dict(cmap='jet', interpolation='nearest'))

    def test_plot_options_two_object(self):
        im = Image(np.random.rand(10, 10))
        imopts1 = opts.Image(interpolation='nearest')
        imopts2 = opts.Image(cmap='hsv')
        styled_im = im.options(imopts1, imopts2)
        self.assertEqual(self.lookup_options(im, 'plot').options, {})
        self.assertEqual(self.lookup_options(styled_im, 'style').options, dict(cmap='hsv', interpolation='nearest'))

    def test_plot_options_object_list(self):
        im = Image(np.random.rand(10, 10))
        imopts1 = opts.Image(interpolation='nearest')
        imopts2 = opts.Image(cmap='summer')
        styled_im = im.options([imopts1, imopts2])
        self.assertEqual(self.lookup_options(im, 'plot').options, {})
        self.assertEqual(self.lookup_options(styled_im, 'style').options, dict(cmap='summer', interpolation='nearest'))