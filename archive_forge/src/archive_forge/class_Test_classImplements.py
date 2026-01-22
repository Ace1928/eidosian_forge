import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
class Test_classImplements(_ImplementsTestMixin, unittest.TestCase):

    def _callFUT(self, cls, iface):
        from zope.interface.declarations import classImplements
        result = classImplements(cls, iface)
        self.assertIsNone(result)
        return cls

    def __check_implementer_redundant(self, Base):
        Base, IBase = self._check_implementer(Base)

        class Child(Base):
            pass
        returned = self._callFUT(Child, IBase)
        self.assertIn('__implemented__', returned.__dict__)
        self.assertNotIn('__providedBy__', returned.__dict__)
        self.assertIn('__provides__', returned.__dict__)
        spec = Child.__implemented__
        self.assertEqual(spec.declared, ())
        self.assertEqual(spec.inherit, Child)
        self.assertTrue(IBase.providedBy(Child()))

    def test_redundant_implementer_empty_class_declarations(self):

        class Foo:
            pass
        self.__check_implementer_redundant(Foo)

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

    def _order_for_two(self, applied_first, applied_second):
        return (applied_first, applied_second)

    def test_w_existing_Implements(self):
        from zope.interface.declarations import Implements
        from zope.interface.interface import InterfaceClass
        IFoo = InterfaceClass('IFoo')
        IBar = InterfaceClass('IBar')
        impl = Implements(IFoo)
        impl.declared = (IFoo,)

        class Foo:
            __implemented__ = impl
        impl.inherit = Foo
        self._callFUT(Foo, IBar)
        self.assertIs(Foo.__implemented__, impl)
        self.assertEqual(impl.inherit, Foo)
        self.assertEqual(impl.declared, self._order_for_two(IFoo, IBar))

    def test_w_existing_Implements_w_bases(self):
        from zope.interface.declarations import Implements
        from zope.interface.interface import InterfaceClass
        IRoot = InterfaceClass('IRoot')
        ISecondRoot = InterfaceClass('ISecondRoot')
        IExtendsRoot = InterfaceClass('IExtendsRoot', (IRoot,))
        impl_root = Implements.named('Root', IRoot)
        impl_root.declared = (IRoot,)

        class Root1:
            __implemented__ = impl_root

        class Root2:
            __implemented__ = impl_root
        impl_extends_root = Implements.named('ExtendsRoot1', IExtendsRoot)
        impl_extends_root.declared = (IExtendsRoot,)

        class ExtendsRoot(Root1, Root2):
            __implemented__ = impl_extends_root
        impl_extends_root.inherit = ExtendsRoot
        self._callFUT(ExtendsRoot, ISecondRoot)
        self.assertIs(ExtendsRoot.__implemented__, impl_extends_root)
        self.assertEqual(impl_extends_root.inherit, ExtendsRoot)
        self.assertEqual(impl_extends_root.declared, self._order_for_two(IExtendsRoot, ISecondRoot))
        self.assertEqual(impl_extends_root.__bases__, self._order_for_two(IExtendsRoot, ISecondRoot) + (impl_root,))