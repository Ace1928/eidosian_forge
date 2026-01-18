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
def test_regrid_upsampling(self):
    img = Image(([0.5, 1.5], [0.5, 1.5], [[0, 1], [2, 3]]))
    regridded = regrid(img, width=4, height=4, upsample=True, dynamic=False)
    expected = Image(([0.25, 0.75, 1.25, 1.75], [0.25, 0.75, 1.25, 1.75], [[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 3, 3], [2, 2, 3, 3]]))
    self.assertEqual(regridded, expected)