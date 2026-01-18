import os
import signal
from testtools.helpers import try_import
from testtools import skipIf
from testtools.matchers import (
from ._helpers import NeedsTwistedTestCase
@skipIf(os.name != 'posix', 'Sending SIGINT with os.kill is posix only')
def test_sigint_raises_no_result_error(self):
    SIGINT = getattr(signal, 'SIGINT', None)
    if not SIGINT:
        self.skipTest('SIGINT not available')
    reactor = self.make_reactor()
    spinner = self.make_spinner(reactor)
    timeout = self.make_timeout()
    reactor.callLater(timeout, os.kill, os.getpid(), SIGINT)
    self.assertThat(lambda: spinner.run(timeout * 5, defer.Deferred), Raises(MatchesException(_spinner.NoResultError)))
    self.assertEqual([], spinner._clean())