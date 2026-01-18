from testtools.helpers import try_import
from testtools.matchers import (
from ._helpers import NeedsTwistedTestCase
def test_not_fired(self):
    self.assertThat(lambda: extract_result(defer.Deferred()), Raises(MatchesException(DeferredNotFired)))