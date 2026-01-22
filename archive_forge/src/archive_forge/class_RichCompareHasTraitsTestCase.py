import unittest
import warnings
from traits.api import (
class RichCompareHasTraitsTestCase(unittest.TestCase, RichCompareTests):

    def setUp(self):
        self.a = Foo(name='a')
        self.same_as_a = Foo(name='a')
        self.different_from_a = Foo(name='not a')

    def test_assumptions(self):
        self.assertIsNot(self.a, self.same_as_a)
        self.assertIsNot(self.a, self.different_from_a)
        self.assertEqual(self.a.name, self.same_as_a.name)
        self.assertNotEqual(self.a.name, self.different_from_a.name)