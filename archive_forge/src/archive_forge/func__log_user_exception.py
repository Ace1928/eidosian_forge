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
def _log_user_exception(self, e):
    """Raise 'e' and report it as a user exception."""
    try:
        raise e
    except e.__class__:
        self._got_user_exception(sys.exc_info())