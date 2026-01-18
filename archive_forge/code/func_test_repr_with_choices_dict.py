import re
import unittest
from oslo_config import types
def test_repr_with_choices_dict(self):
    t = types.Integer(choices=[(80, 'ab'), (457, 'xy')])
    self.assertEqual('Integer(choices=[80, 457])', repr(t))