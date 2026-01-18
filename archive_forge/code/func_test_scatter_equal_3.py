import numpy as np
from holoviews import Bars, Curve, Dimension, Histogram, Points, Scatter, VectorField
from holoviews.element.comparison import ComparisonTestCase
def test_scatter_equal_3(self):
    self.assertEqual(self.scatter3, self.scatter3)