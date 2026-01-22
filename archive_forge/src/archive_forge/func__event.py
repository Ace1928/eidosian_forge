import datetime
from io import BytesIO
from testtools import TestCase
from testtools.matchers import Contains, HasLength
from testtools.testresult.doubles import StreamResult
from testtools.tests.test_testresult import TestStreamResultContract
import subunit
import iso8601
def _event(self, test_status=None, test_id=None, route_code=None, timestamp=None, tags=None, mime_type=None, file_name=None, file_bytes=None, eof=False, runnable=True):
    return ('status', test_id, test_status, tags, runnable, file_name, file_bytes, eof, mime_type, route_code, timestamp)