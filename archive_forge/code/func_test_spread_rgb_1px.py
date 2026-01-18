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
def test_spread_rgb_1px(self):
    arr = np.array([[[0, 0, 0], [0, 1, 1], [0, 1, 1]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]], dtype=np.uint8).T * 255
    spreaded = spread(RGB(arr))
    arr = np.array([[[0, 0, 1], [0, 0, 1], [0, 0, 1]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[1, 1, 1], [1, 1, 1], [1, 1, 1]]], dtype=np.uint8).T * 255
    self.assertEqual(spreaded, RGB(arr))