import unittest
from traits.api import (
def arg_check3(self, object, name, new):
    self.calls += 1
    self.tc.assertIs(object, self.exp_object)
    self.tc.assertEqual(name, self.exp_name)
    self.tc.assertEqual(new, self.exp_new)