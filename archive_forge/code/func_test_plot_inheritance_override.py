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
def test_plot_inheritance_override(self):
    """Overriding an element"""
    hist2 = opts.apply_groups(self.hist, plot={'plot1': 'plot_child'})
    self.assertEqual(self.lookup_options(hist2, 'plot').options, dict(plot1='plot_child', plot2='plot2'))
    self.assertEqual(self.lookup_options(hist2, 'style').options, self.default_style)