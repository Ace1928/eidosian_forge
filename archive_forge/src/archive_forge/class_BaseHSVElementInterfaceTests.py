import datetime as dt
from unittest import SkipTest
import numpy as np
from holoviews import HSV, RGB, Curve, Dataset, Dimension, Image, Table
from holoviews.core.data.interface import DataError
from holoviews.core.util import date_range
from .base import DatatypeContext, GriddedInterfaceTests, InterfaceTests
class BaseHSVElementInterfaceTests(InterfaceTests):
    element = HSV
    __test__ = False

    def init_grid_data(self):
        self.xs = np.linspace(-9, 9, 3)
        self.ys = np.linspace(0.5, 9.5, 3)
        self.hsv_array = np.zeros((3, 3, 3))
        self.hsv_array[0, 0] = 1

    def init_data(self):
        self.hsv = HSV(self.hsv_array[::-1], bounds=(-10, 0, 10, 10))

    def test_hsv_rgb_interface(self):
        R = self.hsv.rgb[..., 'R'].dimension_values(2, expanded=False, flat=False)
        G = self.hsv.rgb[..., 'G'].dimension_values(2, expanded=False, flat=False)
        B = self.hsv.rgb[..., 'B'].dimension_values(2, expanded=False, flat=False)
        self.assertEqual(R[0, 0], 1)
        self.assertEqual(G[0, 0], 0)
        self.assertEqual(B[0, 0], 0)