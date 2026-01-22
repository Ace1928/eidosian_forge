import re
from testtools import TestCase
from testtools.compat import (
from testtools.matchers._basic import (
from testtools.tests.helpers import FullStackRunTest
from testtools.tests.matchers.helpers import TestMatchersInterface
class StartsWithTests(TestCase):
    run_tests_with = FullStackRunTest

    def test_str(self):
        matcher = StartsWith('bar')
        self.assertEqual("StartsWith('bar')", str(matcher))

    def test_str_with_bytes(self):
        b = _b('§')
        matcher = StartsWith(b)
        self.assertEqual(f'StartsWith({b!r})', str(matcher))

    def test_str_with_unicode(self):
        u = '§'
        matcher = StartsWith(u)
        self.assertEqual(f'StartsWith({u!r})', str(matcher))

    def test_match(self):
        matcher = StartsWith('bar')
        self.assertIs(None, matcher.match('barf'))

    def test_mismatch_returns_does_not_start_with(self):
        matcher = StartsWith('bar')
        self.assertIsInstance(matcher.match('foo'), DoesNotStartWith)

    def test_mismatch_sets_matchee(self):
        matcher = StartsWith('bar')
        mismatch = matcher.match('foo')
        self.assertEqual('foo', mismatch.matchee)

    def test_mismatch_sets_expected(self):
        matcher = StartsWith('bar')
        mismatch = matcher.match('foo')
        self.assertEqual('bar', mismatch.expected)