import re
import unittest
from oslo_config import types
def test_not_equal_with_non_equal_custom_item_types(self):
    it1 = types.Integer()
    it2 = types.String()
    self.assertFalse(it1 == it2)
    self.assertFalse(types.Dict(it1) == types.Dict(it2))