from testtools import (
from testtools.compat import (
from testtools.matchers import (
from testtools.matchers._impl import (
from testtools.tests.helpers import FullStackRunTest
def test_verbose_unicode(self):
    matchee = '§'
    matcher = Equals('a')
    mismatch = matcher.match(matchee)
    expected = 'Match failed. Matchee: %s\nMatcher: %s\nDifference: %s\n' % (text_repr(matchee), matcher, mismatch.describe())
    e = MismatchError(matchee, matcher, mismatch, True)
    self.assertEqual(expected, str(e))