import re
import unittest
from oslo_config import types
def test_with_choices_dict(self):
    t = types.Port(choices=[(80, 'ab'), (457, 'xy')])
    self._test_with_choices(t)