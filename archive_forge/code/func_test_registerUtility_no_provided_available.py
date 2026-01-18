import unittest
from zope.interface import Interface
from zope.interface.adapter import VerifyingAdapterRegistry
from zope.interface.registry import Components
def test_registerUtility_no_provided_available(self):

    class Foo:
        pass
    _info = 'info'
    _name = 'name'
    _to_reg = Foo()
    comp = self._makeOne()
    self.assertRaises(TypeError, comp.registerUtility, _to_reg, None, _name, _info)