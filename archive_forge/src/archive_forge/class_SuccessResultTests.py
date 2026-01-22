from testtools.content import TracebackContent
from testtools.matchers import (
from ._helpers import NeedsTwistedTestCase
from testtools.twistedsupport import has_no_result, failed, succeeded
from twisted.internet import defer
from twisted.python.failure import Failure
class SuccessResultTests(NeedsTwistedTestCase):

    def match(self, matcher, value):
        return succeeded(matcher).match(value)

    def test_succeeded_result_passes(self):
        result = object()
        deferred = defer.succeed(result)
        self.assertThat(self.match(Is(result), deferred), Is(None))

    def test_different_succeeded_result_fails(self):
        result = object()
        deferred = defer.succeed(result)
        matcher = Is(None)
        mismatch = matcher.match(result)
        self.assertThat(self.match(matcher, deferred), mismatches(Equals(mismatch.describe()), Equals(mismatch.get_details())))

    def test_not_fired_fails(self):
        deferred = defer.Deferred()
        arbitrary_matcher = Is(None)
        self.assertThat(self.match(arbitrary_matcher, deferred), mismatches(Equals('Success result expected on %r, found no result instead' % (deferred,))))

    def test_failing_fails(self):
        deferred = defer.Deferred()
        fail = make_failure(RuntimeError('arbitrary failure'))
        deferred.errback(fail)
        arbitrary_matcher = Is(None)
        self.assertThat(self.match(arbitrary_matcher, deferred), mismatches(Equals('Success result expected on %r, found failure result instead: %r' % (deferred, fail)), Equals({'traceback': TracebackContent((fail.type, fail.value, fail.getTracebackObject()), None)})))