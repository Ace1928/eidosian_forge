from .. import units as pq
from ..quantity import Quantity
from ..uncertainquantity import UncertainQuantity
from .common import TestCase
import numpy as np
def test_uncertainquantity_multiply(self):
    a = UncertainQuantity([1, 2], 'm', [0.1, 0.2])
    self.assertQuantityEqual(a * a, [1.0, 4.0] * pq.m ** 2)
    self.assertQuantityEqual((a * a).uncertainty, [0.14142, 0.56568] * pq.m ** 2)
    self.assertQuantityEqual(a * 2, [2, 4] * pq.m)
    self.assertQuantityEqual((a * 2).uncertainty, [0.2, 0.4] * pq.m)