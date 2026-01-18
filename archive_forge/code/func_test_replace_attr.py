from unittest import TestCase
from fastimport.helpers import (
from fastimport import (
def test_replace_attr(self):
    c2 = self.c.copy(mark=b'ccc')
    self.assertEqual(bytes(self.c).replace(b'mark :bbb', b'mark :ccc'), bytes(c2))