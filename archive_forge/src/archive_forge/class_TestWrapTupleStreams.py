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
class TestWrapTupleStreams(unittest.TestCase):

    def test_no_streams(self):
        result = wrap_tuple_streams((1, 2), [], [])
        self.assertEqual(result, (1, 2))

    def test_no_streams_two_kdims(self):
        result = wrap_tuple_streams((1, 2), [Dimension('x'), Dimension('y')], [])
        self.assertEqual(result, (1, 2))

    def test_no_streams_none_value(self):
        result = wrap_tuple_streams((1, None), [Dimension('x'), Dimension('y')], [])
        self.assertEqual(result, (1, None))

    def test_no_streams_one_stream_substitution(self):
        result = wrap_tuple_streams((None, 3), [Dimension('x'), Dimension('y')], [PointerXY(x=-5, y=10)])
        self.assertEqual(result, (-5, 3))

    def test_no_streams_two_stream_substitution(self):
        result = wrap_tuple_streams((None, None), [Dimension('x'), Dimension('y')], [PointerXY(x=0, y=5)])
        self.assertEqual(result, (0, 5))