import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
class Test_directlyProvidedBy(unittest.TestCase):

    def _callFUT(self, *args, **kw):
        from zope.interface.declarations import directlyProvidedBy
        return directlyProvidedBy(*args, **kw)

    def test_wo_declarations_in_class_or_instance(self):

        class Foo:
            pass
        foo = Foo()
        self.assertEqual(list(self._callFUT(foo)), [])

    def test_w_declarations_in_class_but_not_instance(self):
        from zope.interface.declarations import implementer
        from zope.interface.interface import InterfaceClass
        IFoo = InterfaceClass('IFoo')

        @implementer(IFoo)
        class Foo:
            pass
        foo = Foo()
        self.assertEqual(list(self._callFUT(foo)), [])

    def test_w_declarations_in_instance_but_not_class(self):
        from zope.interface.declarations import directlyProvides
        from zope.interface.interface import InterfaceClass
        IFoo = InterfaceClass('IFoo')

        class Foo:
            pass
        foo = Foo()
        directlyProvides(foo, IFoo)
        self.assertEqual(list(self._callFUT(foo)), [IFoo])

    def test_w_declarations_in_instance_and_class(self):
        from zope.interface.declarations import directlyProvides
        from zope.interface.declarations import implementer
        from zope.interface.interface import InterfaceClass
        IFoo = InterfaceClass('IFoo')
        IBar = InterfaceClass('IBar')

        @implementer(IFoo)
        class Foo:
            pass
        foo = Foo()
        directlyProvides(foo, IBar)
        self.assertEqual(list(self._callFUT(foo)), [IBar])