import numpy as np
from holoviews import Bars, Curve, Dimension, Histogram, Points, Scatter, VectorField
from holoviews.element.comparison import ComparisonTestCase
def test_scatter_unequal_data_values(self):
    try:
        self.assertEqual(self.scatter1, self.scatter3)
    except AssertionError as e:
        if not str(e).startswith('Scatter not almost equal to 6 decimals'):
            raise self.failureException('Scatter data mismatch error not raised.')