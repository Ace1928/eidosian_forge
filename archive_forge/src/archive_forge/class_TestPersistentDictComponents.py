import unittest
from zope.interface import Interface
from zope.interface.adapter import VerifyingAdapterRegistry
from zope.interface.registry import Components
class TestPersistentDictComponents(TestPersistentComponents):

    def _getTargetClass(self):
        return PersistentDictComponents

    def _makeOne(self):
        comp = self._getTargetClass()(name='test')
        comp['key'] = 42
        return comp

    def _check_equality_after_pickle(self, made):
        self.assertIn('key', made)
        self.assertEqual(made['key'], 42)