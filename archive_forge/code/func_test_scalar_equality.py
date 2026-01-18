import operator as op
from .. import units as pq
from .common import TestCase
def test_scalar_equality(self):
    self.assertEqual(pq.J == pq.J, [True])
    self.assertEqual(1 * pq.J == pq.J, [True])
    self.assertEqual(str(1 * pq.J) == '1.0 J', True)
    self.assertEqual(pq.J == pq.kg * pq.m ** 2 / pq.s ** 2, [True])
    self.assertEqual(pq.J == pq.erg, [False])
    self.assertEqual(2 * pq.J == pq.J, [False])
    self.assertEqual(pq.J == 2 * pq.kg * pq.m ** 2 / pq.s ** 2, [False])
    self.assertEqual(pq.J == pq.kg, [False])