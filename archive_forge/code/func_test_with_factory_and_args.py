import unittest
from traits.has_traits import HasTraits
from traits.trait_types import Any
def test_with_factory_and_args(self):

    def factory(*args, **kw):
        return ('received', args, kw)
    args = (21, 34, 'some string')
    kw = {'bar': 57}

    class A(HasTraits):
        foo = Any(factory=factory, args=args, kw=kw)
    a = A()
    self.assertEqual(a.foo, ('received', args, kw))