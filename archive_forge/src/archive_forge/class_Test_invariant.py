import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
class Test_invariant(unittest.TestCase):

    def test_w_single(self):
        from zope.interface.interface import TAGGED_DATA
        from zope.interface.interface import invariant

        def _check(*args, **kw):
            raise NotImplementedError()

        class Foo:
            invariant(_check)
        self.assertEqual(getattr(Foo, TAGGED_DATA, None), {'invariants': [_check]})

    def test_w_multiple(self):
        from zope.interface.interface import TAGGED_DATA
        from zope.interface.interface import invariant

        def _check(*args, **kw):
            raise NotImplementedError()

        def _another_check(*args, **kw):
            raise NotImplementedError()

        class Foo:
            invariant(_check)
            invariant(_another_check)
        self.assertEqual(getattr(Foo, TAGGED_DATA, None), {'invariants': [_check, _another_check]})