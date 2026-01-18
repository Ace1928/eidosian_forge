import unittest
from traits.api import (
def test_pattern_list13(self):
    c = Complex(tc=self)
    self.check_complex(c, c.ref, 'ref.[int+test,tint-test]', ['int1', 'int3', 'tint1', 'tint3'], ['int2', 'tint2'])