import pyomo.common.unittest as unittest
from pyomo.environ import (
def test_shortnamelabeler_prefix(self):
    m = self.m
    lbl = ShortNameLabeler(20, '_', prefix='s_', caseInsensitive=True)
    self.assertEqual(lbl(m.mycomp), 'mycomp')
    self.assertEqual(lbl(m.that), 'that')
    self.assertEqual(lbl(self.long1), 's_ngcomponentname_1_')
    self.assertEqual(lbl(self.long2), 's_ntnamerighthere_2_')
    self.assertEqual(lbl(self.long3), 's_onebutdifferent_3_')
    self.assertEqual(lbl(self.long4), 's_ngcomponentname_4_')
    self.assertEqual(lbl(self.long5), 'longcomponentname_1_')
    self.assertEqual(lbl(m.myblock), 'myblock')
    self.assertEqual(lbl(m.myblock.mystreet), 'myblock_mystreet')
    self.assertEqual(lbl(m.ind[3]), 'ind_3_')
    self.assertEqual(lbl(m.ind[10]), 'ind_10_')
    self.assertEqual(lbl(m.ind[1]), 'ind_1_')
    self.assertEqual(lbl(self.thecopy), '_myblock_mystreet_')
    m._myblock = Block()
    m._myblock.mystreet_ = Var()
    self.assertEqual(lbl(m.mycomp), 's_mycomp_5_')
    self.assertEqual(lbl(m._myblock.mystreet_), 's_block_mystreet__6_')
    self.assertEqual(lbl(m.MyComp), 's_MyComp_7_')