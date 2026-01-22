import io
import warnings
import sys
from fixtures import CompoundFixture, Fixture
from testtools.content import Content, text_content
from testtools.content_type import UTF8_TEXT
from testtools.runtest import RunTest, _raise_force_fail_error
from ._deferred import extract_result
from ._spinner import (
from twisted.internet import defer
from twisted.python import log
def _get_log_fixture(self):
    """Return the log fixture we're configured to use."""
    fixtures = []
    if self._suppress_twisted_logging:
        fixtures.append(_NoTwistedLogObservers())
    if self._store_twisted_logs:
        fixtures.append(CaptureTwistedLogs())
    return CompoundFixture(fixtures)