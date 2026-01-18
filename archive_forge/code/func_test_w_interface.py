import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test_w_interface(self):
    from zope.interface.interface import InterfaceClass

    class IFoo(InterfaceClass):
        pass

    def _func():
        """DOCSTRING"""
    method = self._callFUT(_func, interface=IFoo)
    self.assertEqual(method.interface, IFoo)