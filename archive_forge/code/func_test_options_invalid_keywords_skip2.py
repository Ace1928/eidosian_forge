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
def test_options_invalid_keywords_skip2(self):
    with options_policy(skip_invalid=True, warn_on_skip=False):
        opts = Options('test', allowed_keywords=['kw1'], kw1='value', kw2='val')
    self.assertEqual(opts.kwargs, {'kw1': 'value'})