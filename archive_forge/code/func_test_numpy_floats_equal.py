import numpy as np
from holoviews.core import BoundingBox
from holoviews.element.comparison import ComparisonTestCase
def test_numpy_floats_equal(self):
    self.assertEqual(np.float32(3.5), np.float32(3.5))