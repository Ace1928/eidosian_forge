import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
def test_wo_existing_provides(self):
    from zope.interface.interface import InterfaceClass
    IFoo = InterfaceClass('IFoo')

    class Foo:
        pass
    obj = Foo()
    self._callFUT(obj, IFoo)
    self.assertEqual(list(obj.__provides__), [])