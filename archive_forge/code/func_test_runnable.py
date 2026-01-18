import datetime
from io import BytesIO
from testtools import TestCase
from testtools.matchers import Contains, HasLength
from testtools.testresult.doubles import StreamResult
from testtools.tests.test_testresult import TestStreamResultContract
import subunit
import iso8601
def test_runnable(self):
    self.check_event(CONSTANT_RUNNABLE, test_status='success', runnable=False)