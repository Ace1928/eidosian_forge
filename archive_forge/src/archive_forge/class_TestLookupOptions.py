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
class TestLookupOptions(ComparisonTestCase):

    def test_lookup_options_honors_backend(self):
        points = Points([[1, 2], [3, 4]])
        import holoviews.plotting.bokeh
        import holoviews.plotting.mpl
        import holoviews.plotting.plotly
        backends = Store.loaded_backends()
        if 'bokeh' in backends:
            Store.set_current_backend('bokeh')
            if 'matplotlib' in backends:
                options_matplotlib = Store.lookup_options('matplotlib', points, 'style')
            if 'plotly' in backends:
                options_plotly = Store.lookup_options('plotly', points, 'style')
        if 'matplotlib' in backends:
            Store.set_current_backend('matplotlib')
            if 'bokeh' in backends:
                options_bokeh = Store.lookup_options('bokeh', points, 'style')
        if 'matplotlib' in backends:
            for opt in ['cmap', 'color', 'marker']:
                self.assertIn(opt, options_matplotlib.keys())
            self.assertNotIn('muted_alpha', options_matplotlib.keys())
        if 'bokeh' in backends:
            for opt in ['cmap', 'color', 'muted_alpha', 'size']:
                self.assertIn(opt, options_bokeh.keys())
        if 'plotly' in backends:
            for opt in ['color']:
                self.assertIn(opt, options_plotly.keys())
            self.assertNotIn('muted_alpha', options_matplotlib.keys())