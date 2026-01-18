import decimal
import sys
import unittest
from traits.api import Either, HasTraits, Int, CInt, TraitError
from traits.testing.optional_dependencies import numpy, requires_numpy
def test_rejects_dunder_int(self):
    a = A()
    with self.assertRaises(TraitError):
        a.integral = Truncatable()