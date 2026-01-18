from unittest import TestCase
from fastimport.helpers import (
from fastimport import (
def test_simple_copy(self):
    c2 = self.c.copy()
    self.assertFalse(self.c is c2)
    self.assertEqual(bytes(self.c), bytes(c2))