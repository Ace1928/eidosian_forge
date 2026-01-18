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
def test_spread_img_1px(self):
    if ds_version < Version('0.12.0'):
        raise SkipTest('Datashader does not support DataArray yet')
    arr = np.array([[0, 0, 0], [0, 0, 0], [1, 1, 1]]).T
    spreaded = spread(Image(arr))
    arr = np.array([[0, 0, 0], [2, 3, 2], [2, 3, 2]]).T
    self.assertEqual(spreaded, Image(arr))