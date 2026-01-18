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
def test_optiontree_find5(self):
    self.assertEqual(self.options.find('XType.Foo').options('group').options, dict(kw2='value2', kw5='value5'))