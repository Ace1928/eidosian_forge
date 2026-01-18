import datetime
from io import BytesIO
from testtools import TestCase
from testtools.matchers import Contains, HasLength
from testtools.testresult.doubles import StreamResult
from testtools.tests.test_testresult import TestStreamResultContract
import subunit
import iso8601
def test_packet_length_4_word_varint(self):
    packet_data = b'\xb3!@\xc0\x00\x11'
    self.check_events(packet_data, [self._event(test_id='subunit.parser', eof=True, file_name='Packet data', file_bytes=packet_data, mime_type='application/octet-stream'), self._event(test_id='subunit.parser', test_status='fail', eof=True, file_name='Parser Error', file_bytes=b'3 byte maximum given but 4 byte value found.', mime_type='text/plain;charset=utf8')])