import unittest
from traits.api import (
def test_pattern_list4(self):
    c = Complex(tc=self)
    handlers = [c.arg_check0, c.arg_check3, c.arg_check4]
    n = len(handlers)
    pattern = 'ref.[int1,int2,int3]'
    self.multi_register(c, handlers, pattern)
    r0 = c.ref
    r1 = ArgCheckBase()
    c.trait_set(exp_object=c, exp_name='ref', exp_old=r0, exp_new=r1)
    c.ref = r1
    c.trait_set(exp_old=r1, exp_new=r0)
    c.ref = r0
    self.assertEqual(c.calls, 2 * n)
    self.multi_register(c, handlers, pattern, remove=True)
    c.ref = r1
    c.ref = r0
    self.assertEqual(c.calls, 2 * n)