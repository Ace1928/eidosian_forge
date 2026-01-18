import numpy as np
from holoviews.core import BoundingBox
from holoviews.element.comparison import ComparisonTestCase
def test_numpy_floats_unequal(self):
    try:
        self.assertEqual(np.float32(3.5), np.float32(3.51))
    except AssertionError as e:
        if not str(e).startswith('Floats not almost equal to 6 decimals'):
            raise self.failureException('Numpy float mismatch error not raised.')