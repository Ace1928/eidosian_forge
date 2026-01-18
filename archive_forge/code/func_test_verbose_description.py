from testtools import (
from testtools.compat import (
from testtools.matchers import (
from testtools.matchers._impl import (
from testtools.tests.helpers import FullStackRunTest
def test_verbose_description(self):
    matchee = 2
    matcher = Equals(3)
    mismatch = matcher.match(2)
    e = MismatchError(matchee, matcher, mismatch, True)
    expected = 'Match failed. Matchee: %r\nMatcher: %s\nDifference: %s\n' % (matchee, matcher, matcher.match(matchee).describe())
    self.assertEqual(expected, str(e))