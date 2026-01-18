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
def test_regrid_rgb_mean(self):
    arr = (np.arange(10) * np.arange(5)[np.newaxis].T).astype('float64')
    rgb = RGB((range(10), range(5), arr, arr * 2, arr * 2))
    regridded = regrid(rgb, width=2, height=2, dynamic=False)
    new_arr = np.array([[1.6, 5.6], [6.4, 22.4]])
    expected = RGB(([2.0, 7.0], [0.75, 3.25], new_arr, new_arr * 2, new_arr * 2), datatype=['xarray'])
    self.assertEqual(regridded, expected)