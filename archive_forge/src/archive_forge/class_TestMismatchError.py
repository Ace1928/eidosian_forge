from testtools import (
from testtools.compat import (
from testtools.matchers import (
from testtools.matchers._impl import (
from testtools.tests.helpers import FullStackRunTest
class TestMismatchError(TestCase):

    def test_is_assertion_error(self):

        def raise_mismatch_error():
            raise MismatchError(2, Equals(3), Equals(3).match(2))
        self.assertRaises(AssertionError, raise_mismatch_error)

    def test_default_description_is_mismatch(self):
        mismatch = Equals(3).match(2)
        e = MismatchError(2, Equals(3), mismatch)
        self.assertEqual(mismatch.describe(), str(e))

    def test_default_description_unicode(self):
        matchee = '§'
        matcher = Equals('a')
        mismatch = matcher.match(matchee)
        e = MismatchError(matchee, matcher, mismatch)
        self.assertEqual(mismatch.describe(), str(e))

    def test_verbose_description(self):
        matchee = 2
        matcher = Equals(3)
        mismatch = matcher.match(2)
        e = MismatchError(matchee, matcher, mismatch, True)
        expected = 'Match failed. Matchee: %r\nMatcher: %s\nDifference: %s\n' % (matchee, matcher, matcher.match(matchee).describe())
        self.assertEqual(expected, str(e))

    def test_verbose_unicode(self):
        matchee = '§'
        matcher = Equals('a')
        mismatch = matcher.match(matchee)
        expected = 'Match failed. Matchee: %s\nMatcher: %s\nDifference: %s\n' % (text_repr(matchee), matcher, mismatch.describe())
        e = MismatchError(matchee, matcher, mismatch, True)
        self.assertEqual(expected, str(e))