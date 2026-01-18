import datetime
from io import BytesIO
from testtools import TestCase
from testtools.matchers import Contains, HasLength
from testtools.testresult.doubles import StreamResult
from testtools.tests.test_testresult import TestStreamResultContract
import subunit
import iso8601
def test_unknown_status(self):
    result, output = self._make_result()
    self.assertRaises(Exception, result.status, 'foo', 'boo')
    self.assertEqual(b'', output.getvalue())