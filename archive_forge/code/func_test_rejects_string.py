import unittest
from traits.api import BaseFloat, Either, Float, HasTraits, Str, TraitError
from traits.testing.optional_dependencies import numpy, requires_numpy
def test_rejects_string(self):
    a = self.test_class()
    with self.assertRaises(TraitError):
        a.value = '2.3'
    with self.assertRaises(TraitError):
        a.value_or_none = '2.3'