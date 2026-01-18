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
def test_opts_method_dynamicmap_grouped(self):
    dmap = DynamicMap(lambda X: Curve([1, 2, X]), kdims=['X']).redim.range(X=(0, 3))
    retval = dmap.opts(padding=1, clone=True)
    assert retval is not dmap
    self.assertEqual(self.lookup_options(retval[0], 'plot').options, {'padding': 1})