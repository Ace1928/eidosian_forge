from testtools.helpers import try_import
from testtools.matchers import (
from ._helpers import NeedsTwistedTestCase
class TestExtractResult(NeedsTwistedTestCase):
    """Tests for ``extract_result``."""

    def test_not_fired(self):
        self.assertThat(lambda: extract_result(defer.Deferred()), Raises(MatchesException(DeferredNotFired)))

    def test_success(self):
        marker = object()
        d = defer.succeed(marker)
        self.assertThat(extract_result(d), Equals(marker))

    def test_failure(self):
        try:
            1 / 0
        except ZeroDivisionError:
            f = Failure()
        d = defer.fail(f)
        self.assertThat(lambda: extract_result(d), Raises(MatchesException(ZeroDivisionError)))