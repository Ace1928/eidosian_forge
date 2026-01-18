from .. import units as pq
from .common import TestCase
import numpy as np
def test_rescale_integer_argument(self):
    from .. import Quantity
    self.assertQuantityEqual(Quantity(10, pq.deg).rescale(pq.rad), np.pi / 18 * pq.rad)