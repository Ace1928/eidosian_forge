import os
import signal
from testtools.helpers import try_import
from testtools import skipIf
from testtools.matchers import (
from ._helpers import NeedsTwistedTestCase
@skipIf(os.name != 'posix', 'Sending SIGINT with os.kill is posix only')
def test_sigint_raises_no_result_error_second_time(self):
    self.test_sigint_raises_no_result_error()