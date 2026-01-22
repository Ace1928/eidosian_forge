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
class TestCrossIndex(ComparisonTestCase):

    def setUp(self):
        self.values1 = ['A', 'B', 'C']
        self.values2 = [1, 2, 3, 4]
        self.values3 = ['?', '!']
        self.values4 = ['x']

    def test_cross_index_full_product(self):
        values = [self.values1, self.values2, self.values3, self.values4]
        cross_product = list(product(*values))
        for i, p in enumerate(cross_product):
            self.assertEqual(cross_index(values, i), p)

    def test_cross_index_depth_1(self):
        values = [self.values1]
        cross_product = list(product(*values))
        for i, p in enumerate(cross_product):
            self.assertEqual(cross_index(values, i), p)

    def test_cross_index_depth_2(self):
        values = [self.values1, self.values2]
        cross_product = list(product(*values))
        for i, p in enumerate(cross_product):
            self.assertEqual(cross_index(values, i), p)

    def test_cross_index_depth_3(self):
        values = [self.values1, self.values2, self.values3]
        cross_product = list(product(*values))
        for i, p in enumerate(cross_product):
            self.assertEqual(cross_index(values, i), p)

    def test_cross_index_large(self):
        values = [[chr(65 + i) for i in range(26)], list(range(500)), [chr(97 + i) for i in range(26)], [chr(48 + i) for i in range(10)]]
        self.assertEqual(cross_index(values, 50001), ('A', 192, 'i', '1'))
        self.assertEqual(cross_index(values, 500001), ('D', 423, 'c', '1'))