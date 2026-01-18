import os
import signal
from testtools.helpers import try_import
from testtools import skipIf
from testtools.matchers import (
from ._helpers import NeedsTwistedTestCase
def test_unhandled_error(self):
    failures = []

    def make_deferred_but_dont_handle():
        try:
            1 / 0
        except ZeroDivisionError:
            f = Failure()
            failures.append(f)
            defer.fail(f)
    result, errors = _spinner.trap_unhandled_errors(make_deferred_but_dont_handle)
    self.assertIs(None, result)
    self.assertEqual(failures, [error.failResult for error in errors])