from .. import units as pq
from .common import TestCase
def test_units_copy(self):
    self.assertQuantityEqual(pq.m.copy(), pq.m)
    pc_per_cc = pq.CompoundUnit('pc/cm**3')
    self.assertQuantityEqual(pc_per_cc.copy(), pc_per_cc)