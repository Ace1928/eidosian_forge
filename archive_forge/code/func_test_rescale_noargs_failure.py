import unittest
import numpy as np
from .. import units as pq
from .. import quantity
from .common import TestCase
def test_rescale_noargs_failure(self):
    quantity.PREFERRED = [pq.pA]
    q = 10 * pq.V
    try:
        self.assertQuantityEqual(q.rescale_preferred(), q.rescale(pq.mV))
    except:
        self.assertTrue(True)
    else:
        self.assertTrue(False)
    quantity.PREFERRED = []