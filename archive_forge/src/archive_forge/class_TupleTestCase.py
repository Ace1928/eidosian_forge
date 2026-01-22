import unittest
from traits.api import (
from traits.tests.tuple_test_mixin import TupleTestMixin
class TupleTestCase(TupleTestMixin, unittest.TestCase):

    def setUp(self):
        self.trait = Tuple

    def test_unexpected_validation_exceptions_are_propagated(self):

        class A(HasTraits):
            foo = Tuple(BadInt(), BadInt())
            bar = Either(Int, Tuple(BadInt(), BadInt()))
        a = A()
        with self.assertRaises(ZeroDivisionError):
            a.foo = (3, 5)
        with self.assertRaises(ZeroDivisionError):
            a.bar = (3, 5)

    def test_non_constant_defaults(self):

        class A(HasTraits):
            foo = Tuple(List(Int))
        a = A()
        a.foo[0].append(35)
        self.assertEqual(a.foo[0], [35])
        with self.assertRaises(TraitError):
            a.foo[0].append(3.5)
        b = A()
        self.assertEqual(b.foo[0], [])

    def test_constant_defaults(self):

        class A(HasTraits):
            foo = Tuple(Int, Tuple(Str, Int))
        a = A()
        b = A()
        self.assertEqual(a.foo, (0, ('', 0)))
        self.assertIs(a.foo, b.foo)