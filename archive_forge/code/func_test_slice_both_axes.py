import datetime as dt
from unittest import SkipTest
import numpy as np
from holoviews import HSV, RGB, Curve, Dataset, Dimension, Image, Table
from holoviews.core.data.interface import DataError
from holoviews.core.util import date_range
from .base import DatatypeContext, GriddedInterfaceTests, InterfaceTests
def test_slice_both_axes(self):
    sliced = self.rgb[0.3:5.2, 1.2:5.2]
    self.assertEqual(sliced.bounds.lbrt(), (0, 1.0, 6, 5))
    self.assertEqual(sliced.xdensity, 0.5)
    self.assertEqual(sliced.ydensity, 1)
    self.assertEqual(sliced.dimension_values(2, flat=False), self.rgb_array[1:5, 5:8, 0])