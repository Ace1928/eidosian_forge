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
def test_options_valid_keywords3(self):
    opts = Options('test', allowed_keywords=['kw1', 'kw2'], kw1='value1', kw2='value2')
    self.assertEqual(opts.kwargs, {'kw1': 'value1', 'kw2': 'value2'})