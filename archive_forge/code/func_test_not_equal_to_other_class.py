import re
import unittest
from oslo_config import types
def test_not_equal_to_other_class(self):
    self.assertFalse(types.Port() == types.Integer())