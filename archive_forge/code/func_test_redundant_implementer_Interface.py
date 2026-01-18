import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
def test_redundant_implementer_Interface(self):
    from zope.interface import Interface
    from zope.interface import implementedBy
    from zope.interface import ro
    from zope.interface.tests.test_ro import C3Setting

    class Foo:
        pass
    with C3Setting(ro.C3.STRICT_IRO, False):
        self._callFUT(Foo, Interface)
        self.assertEqual(list(implementedBy(Foo)), [Interface])

        class Baz(Foo):
            pass
        self._callFUT(Baz, Interface)
        self.assertEqual(list(implementedBy(Baz)), [Interface])