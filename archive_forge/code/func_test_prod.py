import numpy as np
from .. import units as pq
from .common import TestCase, unittest
def test_prod(self):
    self.assertQuantityEqual(np.prod(self.q), 24 * pq.J ** 4)