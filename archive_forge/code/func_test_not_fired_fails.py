from testtools.content import TracebackContent
from testtools.matchers import (
from ._helpers import NeedsTwistedTestCase
from testtools.twistedsupport import has_no_result, failed, succeeded
from twisted.internet import defer
from twisted.python.failure import Failure
def test_not_fired_fails(self):
    deferred = defer.Deferred()
    arbitrary_matcher = Is(None)
    self.assertThat(self.match(arbitrary_matcher, deferred), mismatches(Equals('Success result expected on %r, found no result instead' % (deferred,))))