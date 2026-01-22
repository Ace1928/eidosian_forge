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
class AsynchronousDeferredRunTestForBrokenTwisted(AsynchronousDeferredRunTest):
    """Test runner that works around Twisted brokenness re reactor junk.

    There are many APIs within Twisted itself where a Deferred fires but
    leaves cleanup work scheduled for the reactor to do.  Arguably, many of
    these are bugs.  This runner iterates the reactor event loop a number of
    times after every test, in order to shake out these buggy-but-commonplace
    events.
    """

    def _make_spinner(self):
        spinner = super()._make_spinner()
        spinner._OBLIGATORY_REACTOR_ITERATIONS = 2
        return spinner