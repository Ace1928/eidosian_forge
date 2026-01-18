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
def test_polys_inspection_1px_mask_miss(self):
    if spatialpandas is None:
        raise SkipTest('Polygon inspect tests require spatialpandas')
    polys = inspect_polygons(self.polysrgb, max_indicators=3, dynamic=False, pixels=1, x=0, y=0)
    self.assertEqual(polys, Polygons([], vdims='z'))