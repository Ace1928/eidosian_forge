import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
def test_nonclass_cannot_assign_attr(self):
    from zope.interface.interface import InterfaceClass
    IFoo = InterfaceClass('IFoo')
    decorator = self._makeOne(IFoo)
    self.assertRaises(TypeError, decorator, object())