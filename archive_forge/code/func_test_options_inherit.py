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
def test_options_inherit(self):
    original_kws = dict(kw2='value', kw3='value')
    opts = Options('test', **original_kws)
    new_kws = dict(kw4='val4', kw5='val5')
    new_opts = opts(**new_kws)
    self.assertEqual(new_opts.options, dict(original_kws, **new_kws))