import numpy as np
import pandas as pd
from holoviews.core import Dimension, Dimensioned
from holoviews.element.comparison import ComparisonTestCase
from ..utils import LoggingComparisonTestCase
def test_simple_label_clone(self):
    dim = Dimension('test')
    self.assertEqual(dim.name, 'test')
    clone = dim.clone(label='label')
    self.assertEqual(clone.name, 'test')
    self.assertEqual(clone.label, 'label')