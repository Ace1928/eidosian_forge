import datetime
from io import BytesIO
from testtools import TestCase
from testtools.matchers import Contains, HasLength
from testtools.testresult.doubles import StreamResult
from testtools.tests.test_testresult import TestStreamResultContract
import subunit
import iso8601
class TestStreamResultToBytes(TestCase):

    def _make_result(self):
        output = BytesIO()
        return (subunit.StreamResultToBytes(output), output)

    def test_numbers(self):
        result = subunit.StreamResultToBytes(BytesIO())
        packet = []
        self.assertRaises(Exception, result._write_number, -1, packet)
        self.assertEqual([], packet)
        result._write_number(0, packet)
        self.assertEqual([b'\x00'], packet)
        del packet[:]
        result._write_number(63, packet)
        self.assertEqual([b'?'], packet)
        del packet[:]
        result._write_number(64, packet)
        self.assertEqual([b'@@'], packet)
        del packet[:]
        result._write_number(16383, packet)
        self.assertEqual([b'\x7f\xff'], packet)
        del packet[:]
        result._write_number(16384, packet)
        self.assertEqual([b'\x80@', b'\x00'], packet)
        del packet[:]
        result._write_number(4194303, packet)
        self.assertEqual([b'\xbf\xff', b'\xff'], packet)
        del packet[:]
        result._write_number(4194304, packet)
        self.assertEqual([b'\xc0@\x00\x00'], packet)
        del packet[:]
        result._write_number(1073741823, packet)
        self.assertEqual([b'\xff\xff\xff\xff'], packet)
        del packet[:]
        self.assertRaises(Exception, result._write_number, 1073741824, packet)
        self.assertEqual([], packet)

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

    def test_trivial_enumeration(self):
        result, output = self._make_result()
        result.status('foo', 'exists')
        self.assertEqual(CONSTANT_ENUM, output.getvalue())

    def test_inprogress(self):
        result, output = self._make_result()
        result.status('foo', 'inprogress')
        self.assertEqual(CONSTANT_INPROGRESS, output.getvalue())

    def test_success(self):
        result, output = self._make_result()
        result.status('foo', 'success')
        self.assertEqual(CONSTANT_SUCCESS, output.getvalue())

    def test_uxsuccess(self):
        result, output = self._make_result()
        result.status('foo', 'uxsuccess')
        self.assertEqual(CONSTANT_UXSUCCESS, output.getvalue())

    def test_skip(self):
        result, output = self._make_result()
        result.status('foo', 'skip')
        self.assertEqual(CONSTANT_SKIP, output.getvalue())

    def test_fail(self):
        result, output = self._make_result()
        result.status('foo', 'fail')
        self.assertEqual(CONSTANT_FAIL, output.getvalue())

    def test_xfail(self):
        result, output = self._make_result()
        result.status('foo', 'xfail')
        self.assertEqual(CONSTANT_XFAIL, output.getvalue())

    def test_unknown_status(self):
        result, output = self._make_result()
        self.assertRaises(Exception, result.status, 'foo', 'boo')
        self.assertEqual(b'', output.getvalue())

    def test_eof(self):
        result, output = self._make_result()
        result.status(eof=True)
        self.assertEqual(CONSTANT_EOF, output.getvalue())

    def test_file_content(self):
        result, output = self._make_result()
        result.status(file_name='barney', file_bytes=b'woo')
        self.assertEqual(CONSTANT_FILE_CONTENT, output.getvalue())

    def test_mime(self):
        result, output = self._make_result()
        result.status(mime_type='application/foo; charset=1')
        self.assertEqual(CONSTANT_MIME, output.getvalue())

    def test_route_code(self):
        result, output = self._make_result()
        result.status(test_id='bar', test_status='success', route_code='source')
        self.assertEqual(CONSTANT_ROUTE_CODE, output.getvalue())

    def test_runnable(self):
        result, output = self._make_result()
        result.status('foo', 'success', runnable=False)
        self.assertEqual(CONSTANT_RUNNABLE, output.getvalue())

    def test_tags(self):
        result, output = self._make_result()
        result.status(test_id='bar', test_tags={'foo', 'bar'})
        self.assertThat(CONSTANT_TAGS, Contains(output.getvalue()))

    def test_timestamp(self):
        timestamp = datetime.datetime(2001, 12, 12, 12, 59, 59, 45, iso8601.UTC)
        result, output = self._make_result()
        result.status(test_id='bar', test_status='success', timestamp=timestamp)
        self.assertEqual(CONSTANT_TIMESTAMP, output.getvalue())