import unittest
from zope.interface import Interface
from zope.interface.adapter import VerifyingAdapterRegistry
from zope.interface.registry import Components
def test_register_unregister_identical_objects_provided(self, identical=True):

    class IFoo(Interface):
        pass
    comp = self._makeOne()
    first = object()
    second = first if identical else object()
    comp.registerUtility(first, provided=IFoo)
    comp.registerUtility(second, provided=IFoo, name='bar')
    self.assertEqual(len(comp.utilities._subscribers), 1)
    self.assertEqual(comp.utilities._subscribers, [{IFoo: {'': (first,) if identical else (first, second)}}])
    self.assertEqual(comp.utilities._provided, {IFoo: 3 if identical else 4})
    res = comp.unregisterUtility(first, provided=IFoo)
    self.assertTrue(res)
    res = comp.unregisterUtility(second, provided=IFoo, name='bar')
    self.assertTrue(res)
    self.assertEqual(comp.utilities._provided, {})
    self.assertEqual(len(comp.utilities._subscribers), 0)