import datetime
import math
import unittest
from itertools import product
import numpy as np
import pandas as pd
from holoviews import Dimension, Element
from holoviews.core.util import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.streams import PointerXY
class TestDimensionRange(unittest.TestCase):
    """
    Tests for dimension_range function.
    """

    def setUp(self):
        self.date_range = (np.datetime64(datetime.datetime(2017, 1, 1)), np.datetime64(datetime.datetime(2017, 1, 2)))
        self.date_range2 = (np.datetime64(datetime.datetime(2016, 12, 31)), np.datetime64(datetime.datetime(2017, 1, 3)))

    def test_dimension_range_date_hard_range(self):
        drange = dimension_range(self.date_range2[0], self.date_range2[1], self.date_range, (None, None))
        self.assertEqual(drange, self.date_range)

    def test_dimension_range_date_soft_range(self):
        drange = dimension_range(self.date_range[0], self.date_range[1], (None, None), self.date_range2)
        self.assertEqual(drange, self.date_range2)