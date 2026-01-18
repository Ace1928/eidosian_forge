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
def test_options_inherit_invalid_keywords(self):
    original_kws = dict(kw2='value', kw3='value')
    opts = Options('test', allowed_keywords=['kw3', 'kw2'], **original_kws)
    new_kws = dict(kw4='val4', kw5='val5')
    try:
        opts(**new_kws)
    except OptionError as e:
        self.assertEqual(str(e), "Invalid option 'kw4', valid options are: ['kw2', 'kw3'].")