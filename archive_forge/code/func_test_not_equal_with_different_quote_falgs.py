import re
import unittest
from oslo_config import types
def test_not_equal_with_different_quote_falgs(self):
    t1 = types.String(quotes=False)
    t2 = types.String(quotes=True)
    self.assertFalse(t1 == t2)