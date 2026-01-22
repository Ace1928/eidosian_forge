import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
class TestProvidesClassRepr(unittest.TestCase):

    def _getTargetClass(self):
        from zope.interface.declarations import ProvidesClass
        return ProvidesClass

    def _makeOne(self, *args, **kw):
        return self._getTargetClass()(*args, **kw)

    def test__repr__(self):
        from zope.interface.interface import InterfaceClass
        IFoo = InterfaceClass('IFoo')
        assert IFoo.__name__ == 'IFoo'
        assert IFoo.__module__ == __name__
        assert repr(IFoo) == '<InterfaceClass {}.IFoo>'.format(__name__)
        IBar = InterfaceClass('IBar')
        inst = self._makeOne(type(self), IFoo, IBar)
        self.assertEqual(repr(inst), 'directlyProvides(TestProvidesClassRepr, IFoo, IBar)')

    def test__repr__module_provides_typical_use(self):
        from zope.interface.tests import dummy
        provides = dummy.__provides__
        self.assertEqual(repr(provides), "directlyProvides(sys.modules['zope.interface.tests.dummy'], IDummyModule)")

    def test__repr__module_after_pickle(self):
        import pickle
        from zope.interface.tests import dummy
        provides = dummy.__provides__
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.assertRaises(pickle.PicklingError):
                pickle.dumps(provides, proto)

    def test__repr__directlyProvides_module(self):
        import sys
        from zope.interface.declarations import alsoProvides
        from zope.interface.declarations import directlyProvides
        from zope.interface.interface import InterfaceClass
        from zope.interface.tests import dummy
        IFoo = InterfaceClass('IFoo')
        IBar = InterfaceClass('IBar')
        orig_provides = dummy.__provides__
        del dummy.__provides__
        self.addCleanup(setattr, dummy, '__provides__', orig_provides)
        directlyProvides(dummy, IFoo)
        provides = dummy.__provides__
        self.assertEqual(repr(provides), "directlyProvides(sys.modules['zope.interface.tests.dummy'], IFoo)")
        alsoProvides(dummy, IBar)
        provides = dummy.__provides__
        self.assertEqual(repr(provides), "directlyProvides(sys.modules['zope.interface.tests.dummy'], IFoo, IBar)")
        my_module = sys.modules[__name__]
        assert not hasattr(my_module, '__provides__')
        directlyProvides(my_module, IFoo, IBar)
        self.addCleanup(delattr, my_module, '__provides__')
        self.assertIs(my_module.__provides__, provides)
        self.assertEqual(repr(provides), "directlyProvides(('zope.interface.tests.dummy', 'zope.interface.tests.test_declarations'), IFoo, IBar)")

    def test__repr__module_provides_cached_shared(self):
        from zope.interface.declarations import ModuleType
        from zope.interface.interface import InterfaceClass
        IFoo = InterfaceClass('IFoo')
        inst = self._makeOne(ModuleType, IFoo)
        inst._v_module_names += ('some.module',)
        inst._v_module_names += ('another.module',)
        self.assertEqual(repr(inst), "directlyProvides(('some.module', 'another.module'), IFoo)")

    def test__repr__duplicate_names(self):
        from zope.interface.interface import InterfaceClass
        IFoo = InterfaceClass('IFoo', __module__='mod1')
        IFoo2 = InterfaceClass('IFoo', __module__='mod2')
        IBaz = InterfaceClass('IBaz')
        inst = self._makeOne(type(self), IFoo, IBaz, IFoo2)
        self.assertEqual(repr(inst), 'directlyProvides(TestProvidesClassRepr, IFoo, IBaz, mod2.IFoo)')

    def test__repr__implementedBy_in_interfaces(self):
        from zope.interface import Interface
        from zope.interface import implementedBy

        class IFoo(Interface):
            """Does nothing"""

        class Bar:
            """Does nothing"""
        impl = implementedBy(type(self))
        inst = self._makeOne(Bar, IFoo, impl)
        self.assertEqual(repr(inst), 'directlyProvides(Bar, IFoo, classImplements(TestProvidesClassRepr))')

    def test__repr__empty_interfaces(self):
        inst = self._makeOne(type(self))
        self.assertEqual(repr(inst), 'directlyProvides(TestProvidesClassRepr)')

    def test__repr__non_class(self):

        class Object:
            __bases__ = ()
            __str__ = lambda _: self.fail('Should not call str')

            def __repr__(self):
                return '<Object>'
        inst = self._makeOne(Object())
        self.assertEqual(repr(inst), 'directlyProvides(<Object>)')

    def test__repr__providedBy_from_class(self):
        from zope.interface.declarations import implementer
        from zope.interface.declarations import providedBy
        from zope.interface.interface import InterfaceClass
        IFoo = InterfaceClass('IFoo')

        @implementer(IFoo)
        class Foo:
            pass
        inst = providedBy(Foo())
        self.assertEqual(repr(inst), 'classImplements(Foo, IFoo)')

    def test__repr__providedBy_alsoProvides(self):
        from zope.interface.declarations import alsoProvides
        from zope.interface.declarations import implementer
        from zope.interface.declarations import providedBy
        from zope.interface.interface import InterfaceClass
        IFoo = InterfaceClass('IFoo')
        IBar = InterfaceClass('IBar')

        @implementer(IFoo)
        class Foo:
            pass
        foo = Foo()
        alsoProvides(foo, IBar)
        inst = providedBy(foo)
        self.assertEqual(repr(inst), 'directlyProvides(Foo, IBar, classImplements(Foo, IFoo))')