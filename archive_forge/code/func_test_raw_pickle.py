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
def test_raw_pickle(self):
    """
        Test usual pickle saving and loading (no style information preserved)
        """
    fname = 'test_raw_pickle.pkl'
    with open(fname, 'wb') as handle:
        pickle.dump(self.raw, handle)
    self.clear_options()
    with open(fname, 'rb') as handle:
        img = pickle.load(handle)
    self.assertEqual(self.raw, img)
    pickle.current_backend = 'matplotlib'
    mpl_opts = Store.lookup_options('matplotlib', img, 'style').options
    self.assertEqual(mpl_opts, {})
    Store.current_backend = 'bokeh'
    bokeh_opts = Store.lookup_options('bokeh', img, 'style').options
    self.assertEqual(bokeh_opts, {})