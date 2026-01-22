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
class TestMaxRange(unittest.TestCase):
    """
    Tests for max_range function.
    """

    def setUp(self):
        self.ranges1 = [(-0.2, 0.5), (0, 1), (-0.37, 1.02), (np.nan, 0.3)]
        self.ranges2 = [(np.nan, np.nan), (np.nan, np.nan)]

    def test_max_range1(self):
        self.assertEqual(max_range(self.ranges1), (-0.37, 1.02))

    def test_max_range2(self):
        lower, upper = max_range(self.ranges2)
        self.assertTrue(math.isnan(lower))
        self.assertTrue(math.isnan(upper))

    def test_max_range3(self):
        periods = [(pd.Period('1990', freq='M'), pd.Period('1991', freq='M'))]
        expected = (np.datetime64('1990', 'ns'), np.datetime64('1991', 'ns'))
        self.assertEqual(max_range(periods), expected)