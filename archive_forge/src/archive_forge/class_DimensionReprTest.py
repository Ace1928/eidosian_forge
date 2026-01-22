import numpy as np
import pandas as pd
from holoviews.core import Dimension, Dimensioned
from holoviews.element.comparison import ComparisonTestCase
from ..utils import LoggingComparisonTestCase
class DimensionReprTest(ComparisonTestCase):

    def test_name_dimension_repr(self):
        dim = Dimension('test')
        self.assertEqual(repr(dim), "Dimension('test')")

    def test_name_dimension_repr_eval_equality(self):
        dim = Dimension('test')
        self.assertEqual(eval(repr(dim)) == dim, True)

    def test_name_dimension_repr_tuple(self):
        dim = Dimension(('test', 'Test Dimension'))
        self.assertEqual(repr(dim), "Dimension('test', label='Test Dimension')")

    def test_name_dimension_repr_tuple_eval_equality(self):
        dim = Dimension(('test', 'Test Dimension'))
        self.assertEqual(eval(repr(dim)) == dim, True)

    def test_name_dimension_repr_params(self):
        dim = Dimension('test', label='Test Dimension', unit='m')
        self.assertEqual(repr(dim), "Dimension('test', label='Test Dimension', unit='m')")

    def test_name_dimension_repr_params_eval_equality(self):
        dim = Dimension('test', label='Test Dimension', unit='m')
        self.assertEqual(eval(repr(dim)) == dim, True)

    def test_pprint_value_boolean(self):
        dim = Dimension('test')
        self.assertEqual(dim.pprint_value(True), 'True')
        self.assertEqual(dim.pprint_value(False), 'False')