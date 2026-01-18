import pathlib
import sys
from unittest import TestCase, SkipTest
import holoviews as hv
import numpy as np
import pandas as pd
import pytest
from hvplot.util import proj_to_cartopy
from packaging.version import Version
def test_points_project_xlim_and_ylim_with_geo(self):
    points = self.cities.hvplot(geo=True, xlim=(-10, 10), ylim=(-20, -10))
    opts = hv.Store.lookup_options('bokeh', points, 'plot').options
    np.testing.assert_allclose(opts['xlim'], (-10, 10))
    np.testing.assert_allclose(opts['ylim'], (-20, -10))