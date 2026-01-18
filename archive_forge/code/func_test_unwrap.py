import numpy as np
from .. import units as pq
from .common import TestCase, unittest
@unittest.expectedFailure
def test_unwrap(self):
    self.assertQuantityEqual(np.unwrap([0, 3 * np.pi] * pq.radians), [0, np.pi])
    self.assertQuantityEqual(np.unwrap([0, 540] * pq.deg), [0, 180] * pq.deg)