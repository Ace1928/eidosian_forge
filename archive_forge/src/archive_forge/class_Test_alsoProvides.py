import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
class Test_alsoProvides(unittest.TestCase):

    def _callFUT(self, *args, **kw):
        from zope.interface.declarations import alsoProvides
        return alsoProvides(*args, **kw)

    def test_wo_existing_provides(self):
        from zope.interface.declarations import ProvidesClass
        from zope.interface.interface import InterfaceClass
        IFoo = InterfaceClass('IFoo')

        class Foo:
            pass
        obj = Foo()
        self._callFUT(obj, IFoo)
        self.assertIsInstance(obj.__provides__, ProvidesClass)
        self.assertEqual(list(obj.__provides__), [IFoo])

    def test_w_existing_provides(self):
        from zope.interface.declarations import ProvidesClass
        from zope.interface.declarations import directlyProvides
        from zope.interface.interface import InterfaceClass
        IFoo = InterfaceClass('IFoo')
        IBar = InterfaceClass('IBar')

        class Foo:
            pass
        obj = Foo()
        directlyProvides(obj, IFoo)
        self._callFUT(obj, IBar)
        self.assertIsInstance(obj.__provides__, ProvidesClass)
        self.assertEqual(list(obj.__provides__), [IFoo, IBar])