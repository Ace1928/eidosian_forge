import unittest
from genshi.core import Attrs, QName
from genshi.input import XML
from genshi.path import Path, PathParser, PathSyntaxError, GenericStrategy, \
from genshi.tests.test_utils import doctest_suite
def test_simple_strategy_support(self):
    self.assertTrue(self._test_support(SimplePathStrategy, 'a/b'))
    self.assertTrue(self._test_support(SimplePathStrategy, 'self::a/b'))
    self.assertTrue(self._test_support(SimplePathStrategy, 'descendant::a/b'))
    self.assertTrue(self._test_support(SimplePathStrategy, 'descendant-or-self::a/b'))
    self.assertTrue(self._test_support(SimplePathStrategy, '//a/b'))
    self.assertTrue(self._test_support(SimplePathStrategy, 'a/@b'))
    self.assertTrue(self._test_support(SimplePathStrategy, 'a/text()'))
    self.assertTrue(not self._test_support(SimplePathStrategy, 'a//b'))
    self.assertTrue(not self._test_support(SimplePathStrategy, 'node()/@a'))
    self.assertTrue(not self._test_support(SimplePathStrategy, '@a'))
    self.assertTrue(not self._test_support(SimplePathStrategy, 'foo:bar'))
    self.assertTrue(not self._test_support(SimplePathStrategy, 'a/@foo:bar'))