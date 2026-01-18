import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
def test_w_classless_object(self):
    from zope.interface.declarations import ProvidesClass
    from zope.interface.interface import InterfaceClass
    IFoo = InterfaceClass('IFoo')
    the_dict = {}

    class Foo:

        def __getattribute__(self, name):
            if name == '__class__':
                return None
            raise NotImplementedError(name)

        def __setattr__(self, name, value):
            the_dict[name] = value
    obj = Foo()
    self._callFUT(obj, IFoo)
    self.assertIsInstance(the_dict['__provides__'], ProvidesClass)
    self.assertEqual(list(the_dict['__provides__']), [IFoo])