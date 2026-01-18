import pyomo.common.unittest as unittest
from pyomo.environ import (
def test_textlabeler(self):
    m = self.m
    lbl = TextLabeler()
    self.assertEqual(lbl(m.mycomp), 'mycomp')
    self.assertEqual(lbl(m.mycomp), 'mycomp')
    self.assertEqual(lbl(m.that), 'that')
    self.assertEqual(lbl(self.long1), 'myverylongcomponentname')
    self.assertEqual(lbl(m.myblock), 'myblock')
    self.assertEqual(lbl(m.myblock.mystreet), 'myblock_mystreet')
    self.assertEqual(lbl(self.thecopy), '_myblock_mystreet_')
    self.assertEqual(lbl(m.ind[3]), 'ind(3)')
    self.assertEqual(lbl(m.ind[10]), 'ind(10)')
    self.assertEqual(lbl(m.ind[1]), 'ind(1)')