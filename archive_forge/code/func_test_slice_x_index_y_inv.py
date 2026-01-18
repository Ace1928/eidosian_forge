import datetime as dt
from itertools import product
from unittest import SkipTest
import numpy as np
import pandas as pd
from holoviews.core.data import Dataset
from holoviews.core.data.interface import DataError
from holoviews.core.util import date_range
from holoviews.element import HSV, RGB, Curve, Image
from holoviews.util.transform import dim
from .base import (
from .test_imageinterface import (
def test_slice_x_index_y_inv(self):
    sliced = self.image_inv[0.3:5.2, 5.2]
    self.assertEqual(sliced.bounds.lbrt(), (0, 5.0, 6.0, 6.0))
    self.assertEqual(sliced.xdensity, 0.5)
    self.assertEqual(sliced.ydensity, 1)
    self.assertEqual(sliced.dimension_values(2, flat=False), self.array[5:6, 5:8])