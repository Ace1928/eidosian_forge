import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
class DeclarationTests(EmptyDeclarationTests):

    def _getTargetClass(self):
        from zope.interface.declarations import Declaration
        return Declaration

    def _makeOne(self, *args, **kw):
        return self._getTargetClass()(*args, **kw)

    def test_ctor_no_bases(self):
        decl = self._makeOne()
        self.assertEqual(list(decl.__bases__), [])

    def test_ctor_w_interface_in_bases(self):
        from zope.interface.interface import InterfaceClass
        IFoo = InterfaceClass('IFoo')
        decl = self._makeOne(IFoo)
        self.assertEqual(list(decl.__bases__), [IFoo])

    def test_ctor_w_implements_in_bases(self):
        from zope.interface.declarations import Implements
        impl = Implements()
        decl = self._makeOne(impl)
        self.assertEqual(list(decl.__bases__), [impl])

    def test_changed_wo_existing__v_attrs(self):
        decl = self._makeOne()
        decl.changed(decl)
        self.assertIsNone(decl._v_attrs)

    def test___contains__w_self(self):
        decl = self._makeOne()
        self.assertNotIn(decl, decl)

    def test___contains__w_unrelated_iface(self):
        from zope.interface.interface import InterfaceClass
        IFoo = InterfaceClass('IFoo')
        decl = self._makeOne()
        self.assertNotIn(IFoo, decl)

    def test___contains__w_base_interface(self):
        from zope.interface.interface import InterfaceClass
        IFoo = InterfaceClass('IFoo')
        decl = self._makeOne(IFoo)
        self.assertIn(IFoo, decl)

    def test___iter___single_base(self):
        from zope.interface.interface import InterfaceClass
        IFoo = InterfaceClass('IFoo')
        decl = self._makeOne(IFoo)
        self.assertEqual(list(decl), [IFoo])

    def test___iter___multiple_bases(self):
        from zope.interface.interface import InterfaceClass
        IFoo = InterfaceClass('IFoo')
        IBar = InterfaceClass('IBar')
        decl = self._makeOne(IFoo, IBar)
        self.assertEqual(list(decl), [IFoo, IBar])

    def test___iter___inheritance(self):
        from zope.interface.interface import InterfaceClass
        IFoo = InterfaceClass('IFoo')
        IBar = InterfaceClass('IBar', (IFoo,))
        decl = self._makeOne(IBar)
        self.assertEqual(list(decl), [IBar])

    def test___iter___w_nested_sequence_overlap(self):
        from zope.interface.interface import InterfaceClass
        IFoo = InterfaceClass('IFoo')
        IBar = InterfaceClass('IBar')
        decl = self._makeOne(IBar, (IFoo, IBar))
        self.assertEqual(list(decl), [IBar, IFoo])

    def test_flattened_single_base(self):
        from zope.interface.interface import Interface
        from zope.interface.interface import InterfaceClass
        IFoo = InterfaceClass('IFoo')
        decl = self._makeOne(IFoo)
        self.assertEqual(list(decl.flattened()), [IFoo, Interface])

    def test_flattened_multiple_bases(self):
        from zope.interface.interface import Interface
        from zope.interface.interface import InterfaceClass
        IFoo = InterfaceClass('IFoo')
        IBar = InterfaceClass('IBar')
        decl = self._makeOne(IFoo, IBar)
        self.assertEqual(list(decl.flattened()), [IFoo, IBar, Interface])

    def test_flattened_inheritance(self):
        from zope.interface.interface import Interface
        from zope.interface.interface import InterfaceClass
        IFoo = InterfaceClass('IFoo')
        IBar = InterfaceClass('IBar', (IFoo,))
        decl = self._makeOne(IBar)
        self.assertEqual(list(decl.flattened()), [IBar, IFoo, Interface])

    def test_flattened_w_nested_sequence_overlap(self):
        from zope.interface.interface import Interface
        from zope.interface.interface import InterfaceClass
        IFoo = InterfaceClass('IFoo')
        IBar = InterfaceClass('IBar')
        decl = self._makeOne(IBar, (IFoo, IBar))
        self.assertEqual(list(decl.flattened()), [IBar, IFoo, Interface])

    def test___sub___unrelated_interface(self):
        from zope.interface.interface import InterfaceClass
        IFoo = InterfaceClass('IFoo')
        IBar = InterfaceClass('IBar')
        before = self._makeOne(IFoo)
        after = before - IBar
        self.assertIsInstance(after, self._getTargetClass())
        self.assertEqual(list(after), [IFoo])

    def test___sub___related_interface(self):
        from zope.interface.interface import InterfaceClass
        IFoo = InterfaceClass('IFoo')
        before = self._makeOne(IFoo)
        after = before - IFoo
        self.assertEqual(list(after), [])

    def test___sub___related_interface_by_inheritance(self):
        from zope.interface.interface import InterfaceClass
        IFoo = InterfaceClass('IFoo')
        IBar = InterfaceClass('IBar', (IFoo,))
        before = self._makeOne(IBar)
        after = before - IBar
        self.assertEqual(list(after), [])

    def test___add___unrelated_interface(self):
        from zope.interface.interface import InterfaceClass
        IFoo = InterfaceClass('IFoo')
        IBar = InterfaceClass('IBar')
        before = self._makeOne(IFoo)
        after = before + IBar
        self.assertIsInstance(after, self._getTargetClass())
        self.assertEqual(list(after), [IFoo, IBar])

    def test___add___related_interface(self):
        from zope.interface.interface import InterfaceClass
        IFoo = InterfaceClass('IFoo')
        IBar = InterfaceClass('IBar')
        IBaz = InterfaceClass('IBaz')
        before = self._makeOne(IFoo, IBar)
        other = self._makeOne(IBar, IBaz)
        after = before + other
        self.assertEqual(list(after), [IFoo, IBar, IBaz])

    def test___add___overlapping_interface(self):
        from zope.interface import Interface
        from zope.interface import ro
        from zope.interface.interface import InterfaceClass
        from zope.interface.tests.test_ro import C3Setting
        IBase = InterfaceClass('IBase')
        IDerived = InterfaceClass('IDerived', (IBase,))
        with C3Setting(ro.C3.STRICT_IRO, True):
            base = self._makeOne(IBase)
            after = base + IDerived
        self.assertEqual(after.__iro__, (IDerived, IBase, Interface))
        self.assertEqual(after.__bases__, (IDerived, IBase))
        self.assertEqual(list(after), [IDerived, IBase])

    def test___add___overlapping_interface_implementedBy(self):
        from zope.interface import Interface
        from zope.interface import implementedBy
        from zope.interface import implementer
        from zope.interface import ro
        from zope.interface.tests.test_ro import C3Setting

        class IBase(Interface):
            pass

        class IDerived(IBase):
            pass

        @implementer(IBase)
        class Base:
            pass
        with C3Setting(ro.C3.STRICT_IRO, True):
            after = implementedBy(Base) + IDerived
        self.assertEqual(after.__sro__, (after, IDerived, IBase, Interface))
        self.assertEqual(after.__bases__, (IDerived, IBase))
        self.assertEqual(list(after), [IDerived, IBase])