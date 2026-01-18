import datetime
from io import BytesIO
from testtools import TestCase
from testtools.matchers import Contains, HasLength
from testtools.testresult.doubles import StreamResult
from testtools.tests.test_testresult import TestStreamResultContract
import subunit
import iso8601
def test_signature_middle_utf8_char(self):
    utf8_bytes = b'\xe3\xb3\x8a'
    source = BytesIO(utf8_bytes)
    result = StreamResult()
    subunit.ByteStreamToStreamResult(source, non_subunit_name='stdout').run(result)
    self.assertEqual([('status', None, None, None, True, 'stdout', b'\xe3', False, None, None, None), ('status', None, None, None, True, 'stdout', b'\xb3', False, None, None, None), ('status', None, None, None, True, 'stdout', b'\x8a', False, None, None, None)], result._events)