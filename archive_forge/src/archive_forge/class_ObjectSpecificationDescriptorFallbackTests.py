import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
class ObjectSpecificationDescriptorFallbackTests(unittest.TestCase):

    def _getFallbackClass(self):
        from zope.interface.declarations import ObjectSpecificationDescriptorFallback
        return ObjectSpecificationDescriptorFallback
    _getTargetClass = _getFallbackClass

    def _makeOne(self, *args, **kw):
        return self._getTargetClass()(*args, **kw)

    def test_accessed_via_class(self):
        from zope.interface.declarations import Provides
        from zope.interface.interface import InterfaceClass
        IFoo = InterfaceClass('IFoo')

        class Foo:
            pass
        Foo.__provides__ = Provides(Foo, IFoo)
        Foo.__providedBy__ = self._makeOne()
        self.assertEqual(list(Foo.__providedBy__), [IFoo])

    def test_accessed_via_inst_wo_provides(self):
        from zope.interface.declarations import Provides
        from zope.interface.declarations import implementer
        from zope.interface.interface import InterfaceClass
        IFoo = InterfaceClass('IFoo')
        IBar = InterfaceClass('IBar')

        @implementer(IFoo)
        class Foo:
            pass
        Foo.__provides__ = Provides(Foo, IBar)
        Foo.__providedBy__ = self._makeOne()
        foo = Foo()
        self.assertEqual(list(foo.__providedBy__), [IFoo])

    def test_accessed_via_inst_w_provides(self):
        from zope.interface.declarations import Provides
        from zope.interface.declarations import directlyProvides
        from zope.interface.declarations import implementer
        from zope.interface.interface import InterfaceClass
        IFoo = InterfaceClass('IFoo')
        IBar = InterfaceClass('IBar')
        IBaz = InterfaceClass('IBaz')

        @implementer(IFoo)
        class Foo:
            pass
        Foo.__provides__ = Provides(Foo, IBar)
        Foo.__providedBy__ = self._makeOne()
        foo = Foo()
        directlyProvides(foo, IBaz)
        self.assertEqual(list(foo.__providedBy__), [IBaz, IFoo])

    def test_arbitrary_exception_accessing_provides_not_caught(self):

        class MyException(Exception):
            pass

        class Foo:
            __providedBy__ = self._makeOne()

            @property
            def __provides__(self):
                raise MyException
        foo = Foo()
        with self.assertRaises(MyException):
            getattr(foo, '__providedBy__')

    def test_AttributeError_accessing_provides_caught(self):

        class MyException(Exception):
            pass

        class Foo:
            __providedBy__ = self._makeOne()

            @property
            def __provides__(self):
                raise AttributeError
        foo = Foo()
        provided = getattr(foo, '__providedBy__')
        self.assertIsNotNone(provided)

    def test_None_in__provides__overrides(self):
        from zope.interface import Interface
        from zope.interface import implementer

        class IFoo(Interface):
            pass

        @implementer(IFoo)
        class Foo:

            @property
            def __provides__(self):
                return None
        Foo.__providedBy__ = self._makeOne()
        provided = getattr(Foo(), '__providedBy__')
        self.assertIsNone(provided)