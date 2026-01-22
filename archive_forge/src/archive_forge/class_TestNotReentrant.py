import os
import signal
from testtools.helpers import try_import
from testtools import skipIf
from testtools.matchers import (
from ._helpers import NeedsTwistedTestCase
class TestNotReentrant(NeedsTwistedTestCase):

    def test_not_reentrant(self):
        calls = []

        @_spinner.not_reentrant
        def log_something():
            calls.append(None)
            if len(calls) < 5:
                log_something()
        self.assertThat(log_something, Raises(MatchesException(_spinner.ReentryError)))
        self.assertEqual(1, len(calls))

    def test_deeper_stack(self):
        calls = []

        @_spinner.not_reentrant
        def g():
            calls.append(None)
            if len(calls) < 5:
                f()

        @_spinner.not_reentrant
        def f():
            calls.append(None)
            if len(calls) < 5:
                g()
        self.assertThat(f, Raises(MatchesException(_spinner.ReentryError)))
        self.assertEqual(2, len(calls))