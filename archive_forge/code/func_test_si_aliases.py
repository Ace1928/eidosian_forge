import unittest
import numpy as np
from .. import units as pq
from .. import quantity
from .common import TestCase
def test_si_aliases(self):
    prefixes = ['kilo', 'mega', 'giga', 'tera', 'peta', 'exa', 'zetta', 'yotta']
    for prefix in prefixes:
        self.assertQuantityEqual(pq.B.rescale(prefix + 'byte'), pq.B.rescale(prefix + 'bytes'))
        self.assertQuantityEqual(pq.B.rescale(prefix + 'byte'), pq.B.rescale(prefix + 'octet'))
        self.assertQuantityEqual(pq.B.rescale(prefix + 'byte'), pq.B.rescale(prefix + 'octets'))