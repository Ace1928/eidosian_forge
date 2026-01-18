from .. import units as pq
from ..quantity import Quantity
from ..uncertainquantity import UncertainQuantity
from .common import TestCase
import numpy as np
def test_uncertainquantity_negative(self):
    a = UncertainQuantity([1, 2], 'm', [0.1, 0.2])
    self.assertQuantityEqual(-a, [-1.0, -2.0] * pq.m)
    self.assertQuantityEqual((-a).uncertainty, [0.1, 0.2] * pq.m)
    self.assertQuantityEqual(-a, a * -1)
    self.assertQuantityEqual((-a).uncertainty, (a * -1).uncertainty)