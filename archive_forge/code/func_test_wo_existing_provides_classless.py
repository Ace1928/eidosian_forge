import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
def test_wo_existing_provides_classless(self):
    the_dict = {}

    class Foo:

        def __getattribute__(self, name):
            if name == '__class__':
                raise AttributeError(name)
            try:
                return the_dict[name]
            except KeyError:
                raise AttributeError(name)

        def __setattr__(self, name, value):
            raise NotImplementedError()
    foo = Foo()
    spec = self._callFUT(foo)
    self.assertEqual(list(spec), [])