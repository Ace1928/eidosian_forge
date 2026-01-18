import unittest
from traits.api import (
def test_pattern_list5(self):
    c = Complex(tc=self)
    c.on_trait_change(c.arg_check1, 'ref.[int1,int2,int3]')
    self.assertRaises(TraitError, c.trait_set, ref=ArgCheckBase())