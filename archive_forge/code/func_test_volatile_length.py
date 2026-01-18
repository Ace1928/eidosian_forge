import datetime
from io import BytesIO
from testtools import TestCase
from testtools.matchers import Contains, HasLength
from testtools.testresult.doubles import StreamResult
from testtools.tests.test_testresult import TestStreamResultContract
import subunit
import iso8601
def test_volatile_length(self):
    result, output = self._make_result()
    result.status(file_name='', file_bytes=b'\xff' * 0)
    self.assertThat(output.getvalue(), HasLength(10))
    self.assertEqual(b'\n', output.getvalue()[3:4])
    output.seek(0)
    output.truncate()
    result.status(file_name='', file_bytes=b'\xff' * 53)
    self.assertThat(output.getvalue(), HasLength(63))
    self.assertEqual(b'?', output.getvalue()[3:4])
    output.seek(0)
    output.truncate()
    result.status(file_name='', file_bytes=b'\xff' * 54)
    self.assertThat(output.getvalue(), HasLength(65))
    self.assertEqual(b'@A', output.getvalue()[3:5])
    output.seek(0)
    output.truncate()
    result.status(file_name='', file_bytes=b'\xff' * 16371)
    self.assertThat(output.getvalue(), HasLength(16383))
    self.assertEqual(b'\x7f\xff', output.getvalue()[3:5])
    output.seek(0)
    output.truncate()
    result.status(file_name='', file_bytes=b'\xff' * 16372)
    self.assertThat(output.getvalue(), HasLength(16385))
    self.assertEqual(b'\x80@\x01', output.getvalue()[3:6])
    output.seek(0)
    output.truncate()
    result.status(file_name='', file_bytes=b'\xff' * 4194289)
    self.assertThat(output.getvalue(), HasLength(4194303))
    self.assertEqual(b'\xbf\xff\xff', output.getvalue()[3:6])
    output.seek(0)
    output.truncate()
    self.assertRaises(Exception, result.status, file_name='', file_bytes=b'\xff' * 4194290)