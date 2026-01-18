import operator as op
from functools import partial
import sys
import numpy as np
from .. import units as pq
from ..quantity import Quantity
from .common import TestCase
def test_powering(self):
    self.assertQuantityEqual((5.5 * pq.cm) ** 5, 5.5 ** 5 * pq.cm ** 5)
    self.assertQuantityEqual((5.5 * pq.cm) ** 0, 5.5 ** 0 * pq.dimensionless)
    self.assertQuantityEqual((5.5 * pq.J) ** 5, 5.5 ** 5 * pq.J ** 5)
    self.assertQuantityEqual((np.array([1, 2, 3, 4, 5]) * pq.kg) ** 3, np.array([1, 8, 27, 64, 125]) * pq.kg ** 3)

    def q_pow_r(q1, q2):
        return q1 ** q2
    self.assertRaises(ValueError, q_pow_r, 10.0 * pq.m, 10 * pq.J)
    self.assertRaises(ValueError, q_pow_r, 10.0 * pq.m, np.array([1, 2, 3]))
    self.assertQuantityEqual((10 * pq.J) ** (2 * pq.J / pq.J), 100 * pq.J ** 2)
    self.assertRaises(ValueError, q_pow_r, 10.0, 10 * pq.J)
    self.assertQuantityEqual(10 ** (2 * pq.J / pq.J), 100)

    def ipow(q1, q2):
        q1 -= q2
    self.assertRaises(ValueError, ipow, 1 * pq.m, [1, 2])