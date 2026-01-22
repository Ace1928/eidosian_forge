import os
import signal
from testtools import (
from testtools.helpers import try_import
from testtools.matchers import (
from testtools.runtest import RunTest
from testtools.testresult.doubles import ExtendedTestResult
from testtools.tests.helpers import (
from ._helpers import NeedsTwistedTestCase
class ErrorInSetup(Base):
    expected_calls = ['setUp', 'clean-up']
    expected_results = [('addError', RuntimeError)]

    def setUp(self):
        super(X.ErrorInSetup, self).setUp()
        raise RuntimeError('Error in setUp')