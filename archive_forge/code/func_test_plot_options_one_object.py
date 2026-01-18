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
def test_plot_options_one_object(self):
    im = Image(np.random.rand(10, 10))
    imopts = opts.Image(interpolation='nearest', cmap='jet')
    styled_im = im.options(imopts)
    self.assertEqual(self.lookup_options(im, 'plot').options, {})
    self.assertEqual(self.lookup_options(styled_im, 'style').options, dict(cmap='jet', interpolation='nearest'))