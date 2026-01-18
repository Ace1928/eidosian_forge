import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test_verifies(self):
    from zope.interface.interfaces import IElement
    from zope.interface.verify import verifyObject
    element = self._makeOne()
    verifyObject(IElement, element)