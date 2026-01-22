import unittest
from traits.api import (
def arg_check4(self, object, name, old, new):
    self.calls += 1
    self.tc.assertIs(object, self.exp_object)
    self.tc.assertEqual(name, self.exp_name)
    self.tc.assertEqual(old, self.exp_old)
    self.tc.assertEqual(new, self.exp_new)