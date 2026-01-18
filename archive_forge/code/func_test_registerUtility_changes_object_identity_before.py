import unittest
from zope.interface import Interface
from zope.interface.adapter import VerifyingAdapterRegistry
from zope.interface.registry import Components
def test_registerUtility_changes_object_identity_before(self):

    class CompThatChangesAfter2Reg(self._getTargetClass()):
        reg_count = 0

        def registerUtility(self, *args):
            self.reg_count += 1
            if self.reg_count == 2:
                self._utility_registrations = dict(self._utility_registrations)
            super().registerUtility(*args)
    comp = CompThatChangesAfter2Reg()
    comp.registerUtility(object(), Interface)
    self.assertEqual(len(list(comp.registeredUtilities())), 1)

    class IFoo(Interface):
        pass
    comp.registerUtility(object(), IFoo)
    self.assertEqual(len(list(comp.registeredUtilities())), 2)

    class IBar(Interface):
        pass
    comp.registerUtility(object(), IBar)
    self.assertEqual(len(list(comp.registeredUtilities())), 3)