from unittest import SkipTest
import numpy as np
import pandas as pd
from holoviews import Curve, Dataset, Dimension, Distribution, Scatter
from holoviews.core import Apply, Redim
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation import function, histogram
def test_propagate_dataset(self):
    op = function.instance(fn=lambda ds: ds.iloc[:5].clone(dataset=None, pipeline=None))
    new_ds = op(self.ds)
    self.assertEqual(new_ds.dataset, self.ds)