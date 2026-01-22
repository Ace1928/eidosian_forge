import unittest
from traits.api import (
def arg_check2(self, name, new):
    self.calls += 1
    self.tc.assertEqual(name, self.exp_name)
    self.tc.assertEqual(new, self.exp_new)