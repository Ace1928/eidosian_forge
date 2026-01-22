import unittest
from zope.interface import Interface
from zope.interface.adapter import VerifyingAdapterRegistry
from zope.interface.registry import Components
def _check_equality_after_pickle(self, made):
    self.assertIn('key', made)
    self.assertEqual(made['key'], 42)