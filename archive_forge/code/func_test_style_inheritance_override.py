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
def test_style_inheritance_override(self):
    """Overriding an element"""
    hist2 = opts.apply_groups(self.hist, style={'style1': 'style_child'})
    self.assertEqual(self.lookup_options(hist2, 'style').options, dict(style1='style_child', style2='style2'))
    self.assertEqual(self.lookup_options(hist2, 'plot').options, self.default_plot)