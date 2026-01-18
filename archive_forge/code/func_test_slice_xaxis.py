import datetime as dt
from unittest import SkipTest
import numpy as np
from holoviews import HSV, RGB, Curve, Dataset, Dimension, Image, Table
from holoviews.core.data.interface import DataError
from holoviews.core.util import date_range
from .base import DatatypeContext, GriddedInterfaceTests, InterfaceTests
def test_slice_xaxis(self):
    sliced = self.rgb[0.3:5.2]
    self.assertEqual(sliced.bounds.lbrt(), (0, 0, 6, 10))
    self.assertEqual(sliced.xdensity, 0.5)
    self.assertEqual(sliced.ydensity, 1)
    self.assertEqual(sliced.dimension_values(2, flat=False), self.rgb_array[:, 5:8, 0])