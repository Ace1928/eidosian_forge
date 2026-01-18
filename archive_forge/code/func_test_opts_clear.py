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
def test_opts_clear(self):
    im = Image(np.random.rand(10, 10))
    styled_im = opts.apply_groups(im, style=dict(cmap='jet', interpolation='nearest', option1='A', option2='B'), clone=False)
    self.assertEqual(self.lookup_options(im, 'style').options, {'cmap': 'jet', 'interpolation': 'nearest', 'option1': 'A', 'option2': 'B'})
    assert styled_im is im
    cleared = im.opts.clear()
    assert cleared is im
    cleared_options = self.lookup_options(cleared, 'style').options
    self.assertEqual(not any((k in ['option1', 'option2'] for k in cleared_options.keys())), True)