import datetime as dt
from unittest import SkipTest, skipIf
import colorcet as cc
import numpy as np
import pandas as pd
import pytest
from numpy import nan
from packaging.version import Version
from holoviews import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation import apply_when
from holoviews.streams import Tap
from holoviews.util import render
import logging
def test_points_inspection_5px_mask(self):
    points = inspect_points(self.pntsimg, max_indicators=3, dynamic=False, pixels=5, x=-0.1, y=-0.1)
    self.assertEqual(points.dimension_values('x'), np.array([0.2, 0.4, 0]))
    self.assertEqual(points.dimension_values('y'), np.array([0.3, 0.7, 0.99]))