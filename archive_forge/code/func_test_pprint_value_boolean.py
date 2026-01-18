import numpy as np
import pandas as pd
from holoviews.core import Dimension, Dimensioned
from holoviews.element.comparison import ComparisonTestCase
from ..utils import LoggingComparisonTestCase
def test_pprint_value_boolean(self):
    dim = Dimension('test')
    self.assertEqual(dim.pprint_value(True), 'True')
    self.assertEqual(dim.pprint_value(False), 'False')