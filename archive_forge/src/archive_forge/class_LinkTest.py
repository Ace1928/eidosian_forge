import unittest
from traits.api import (
class LinkTest(HasTraits):
    head = Instance(Link)
    calls = Int(0)
    exp_object = Any
    exp_name = Any
    dst_name = Any
    exp_old = Any
    exp_new = Any
    dst_new = Any
    tc = Any

    def arg_check0(self):
        self.calls += 1

    def arg_check1(self, new):
        self.calls += 1
        self.tc.assertEqual(new, self.exp_new)

    def arg_check2(self, name, new):
        self.calls += 1
        self.tc.assertEqual(name, self.exp_name)
        self.tc.assertEqual(new, self.exp_new)

    def arg_check3(self, object, name, new):
        self.calls += 1
        self.tc.assertIs(object, self.exp_object)
        self.tc.assertEqual(name, self.exp_name)
        self.tc.assertEqual(new, self.exp_new)

    def arg_check4(self, object, name, old, new):
        self.calls += 1
        self.tc.assertIs(object, self.exp_object)
        self.tc.assertEqual(name, self.exp_name)
        self.tc.assertEqual(old, self.exp_old)
        self.tc.assertEqual(new, self.exp_new)