from .. import units as pq
from .common import TestCase
def test_units_protected(self):

    def setunits(u, v):
        u.units = v

    def inplace(op, u, val):
        getattr(u, '__i%s__' % op)(val)
    self.assertRaises(AttributeError, setunits, pq.m, pq.ft)
    self.assertRaises(TypeError, inplace, 'add', pq.m, pq.m)
    self.assertRaises(TypeError, inplace, 'sub', pq.m, pq.m)
    self.assertRaises(TypeError, inplace, 'mul', pq.m, pq.m)
    self.assertRaises(TypeError, inplace, 'truediv', pq.m, pq.m)
    self.assertRaises(TypeError, inplace, 'pow', pq.m, 2)