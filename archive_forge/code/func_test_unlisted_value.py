import re
import unittest
from oslo_config import types
def test_unlisted_value(self):
    self.type_instance = types.String(choices=['foo', 'bar'])
    self.assertInvalid('baz')