import numpy as np
import pandas as pd
from holoviews.core import Dimension, Dimensioned
from holoviews.element.comparison import ComparisonTestCase
from ..utils import LoggingComparisonTestCase
def test_simple_str_equality(self):
    dim1 = Dimension('test')
    dim2 = Dimension('test')
    self.assertEqual(dim1 == str(dim2), True)