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
def test_optiontree_find3(self):
    self.assertEqual(self.options.find('MyType.Foo').options('group').options, dict(kw1='value1', kw3='value3'))