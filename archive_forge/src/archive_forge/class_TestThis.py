import unittest
import warnings
from traits.api import (
from traits.testing.optional_dependencies import requires_traitsui
class TestThis(unittest.TestCase):

    def test_this_none(self):
        d = ThisDummy()
        self.assertIsNone(d.allows_none)
        d.allows_none = None
        d.allows_none = ThisDummy()
        self.assertIsNotNone(d.allows_none)
        d.allows_none = None
        self.assertIsNone(d.allows_none)
        self.assertIsNone(d.disallows_none)
        d.disallows_none = ThisDummy()
        self.assertIsNotNone(d.disallows_none)
        with self.assertRaises(TraitError):
            d.disallows_none = None
        self.assertIsNotNone(d.disallows_none)

    def test_this_other_class(self):
        d = ThisDummy()
        with self.assertRaises(TraitError):
            d.allows_none = object()
        self.assertIsNone(d.allows_none)