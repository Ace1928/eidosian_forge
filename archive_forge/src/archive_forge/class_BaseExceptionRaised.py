import os
import signal
from testtools import (
from testtools.helpers import try_import
from testtools.matchers import (
from testtools.runtest import RunTest
from testtools.testresult.doubles import ExtendedTestResult
from testtools.tests.helpers import (
from ._helpers import NeedsTwistedTestCase
class BaseExceptionRaised(Base):
    expected_calls = ['setUp', 'tearDown', 'clean-up']
    expected_results = [('addError', SystemExit)]

    def test_something(self):
        raise SystemExit(0)