import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
class _ImplementsTestMixin:
    FUT_SETS_PROVIDED_BY = True

    def _callFUT(self, cls, iface):
        raise NotImplementedError

    def _check_implementer(self, Foo, orig_spec=None, spec_name=__name__ + '.Foo', inherit='not given'):
        from zope.interface.declarations import ClassProvides
        from zope.interface.interface import InterfaceClass
        IFoo = InterfaceClass('IFoo')
        returned = self._callFUT(Foo, IFoo)
        self.assertIs(returned, Foo)
        spec = Foo.__implemented__
        if orig_spec is not None:
            self.assertIs(spec, orig_spec)
        self.assertEqual(spec.__name__, spec_name)
        inherit = Foo if inherit == 'not given' else inherit
        self.assertIs(spec.inherit, inherit)
        self.assertIs(Foo.__implemented__, spec)
        if self.FUT_SETS_PROVIDED_BY:
            self.assertIsInstance(Foo.__providedBy__, ClassProvides)
            self.assertIsInstance(Foo.__provides__, ClassProvides)
            self.assertEqual(Foo.__provides__, Foo.__providedBy__)
        return (Foo, IFoo)

    def test_class(self):

        class Foo:
            pass
        self._check_implementer(Foo)