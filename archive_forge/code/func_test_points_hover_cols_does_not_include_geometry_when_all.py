import pathlib
import sys
from unittest import TestCase, SkipTest
import holoviews as hv
import numpy as np
import pandas as pd
import pytest
from hvplot.util import proj_to_cartopy
from packaging.version import Version
def test_points_hover_cols_does_not_include_geometry_when_all(self):
    points = self.cities.hvplot(x='x', y='y', hover_cols='all')
    assert points.kdims == ['x', 'y']
    assert points.vdims == ['index', 'name']