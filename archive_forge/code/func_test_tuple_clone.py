import numpy as np
import pandas as pd
from holoviews.core import Dimension, Dimensioned
from holoviews.element.comparison import ComparisonTestCase
from ..utils import LoggingComparisonTestCase
def test_tuple_clone(self):
    dim = Dimension('test')
    self.assertEqual(dim.name, 'test')
    clone = dim.clone(('test', 'A test'))
    self.assertEqual(clone.name, 'test')
    self.assertEqual(clone.label, 'A test')