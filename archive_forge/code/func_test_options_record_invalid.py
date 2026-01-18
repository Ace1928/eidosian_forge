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
def test_options_record_invalid(self):
    StoreOptions.start_recording_skipped()
    with options_policy(skip_invalid=True, warn_on_skip=False):
        Options('test', allowed_keywords=['kw1'], kw1='value', kw2='val')
    errors = StoreOptions.stop_recording_skipped()
    self.assertEqual(len(errors), 1)
    self.assertEqual(errors[0].invalid_keyword, 'kw2')