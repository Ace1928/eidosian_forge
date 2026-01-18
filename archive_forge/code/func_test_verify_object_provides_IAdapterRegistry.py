import unittest
from zope.interface.tests import OptimizationTestMixin
def test_verify_object_provides_IAdapterRegistry(self):
    from zope.interface.interfaces import IAdapterRegistry
    from zope.interface.verify import verifyObject
    registry = self._makeOne()
    verifyObject(IAdapterRegistry, registry)