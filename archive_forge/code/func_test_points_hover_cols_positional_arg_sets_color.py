import pathlib
import sys
from unittest import TestCase, SkipTest
import holoviews as hv
import numpy as np
import pandas as pd
import pytest
from hvplot.util import proj_to_cartopy
from packaging.version import Version
def test_points_hover_cols_positional_arg_sets_color(self):
    points = self.cities.hvplot('name')
    assert points.kdims == ['x', 'y']
    assert points.vdims == ['name']
    opts = hv.Store.lookup_options('bokeh', points, 'style').kwargs
    assert opts['color'] == 'name'