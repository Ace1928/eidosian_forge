import unittest
from zope.interface import Interface
from zope.interface.adapter import VerifyingAdapterRegistry
from zope.interface.registry import Components
def test_registerAdapter_wo_event(self):
    from zope.interface.declarations import InterfaceClass

    class IFoo(InterfaceClass):
        pass
    ifoo = IFoo('IFoo')
    ibar = IFoo('IBar')
    _info = 'info'
    _name = 'name'

    def _factory(context):
        raise NotImplementedError()
    comp = self._makeOne()
    _monkey, _events = self._wrapEvents()
    with _monkey:
        comp.registerAdapter(_factory, (ibar,), ifoo, _name, _info, event=False)
    self.assertEqual(len(_events), 0)