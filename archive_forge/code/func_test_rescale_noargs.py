import unittest
import numpy as np
from .. import units as pq
from .. import quantity
from .common import TestCase
def test_rescale_noargs(self):
    quantity.PREFERRED = [pq.mV, pq.pA]
    q = 10 * pq.V
    self.assertQuantityEqual(q.rescale(), q.rescale(pq.mV))
    q = 5 * pq.A
    self.assertQuantityEqual(q.rescale(), q.rescale(pq.pA))
    quantity.PREFERRED = []