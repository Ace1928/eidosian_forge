import unittest
from io import BytesIO
from testtools.compat import _b
import subunit.chunked
class TestEncode(unittest.TestCase):

    def setUp(self):
        unittest.TestCase.setUp(self)
        self.output = BytesIO()
        self.encoder = subunit.chunked.Encoder(self.output)

    def test_encode_nothing(self):
        self.encoder.close()
        self.assertEqual(_b('0\r\n'), self.output.getvalue())

    def test_encode_empty(self):
        self.encoder.write(_b(''))
        self.encoder.close()
        self.assertEqual(_b('0\r\n'), self.output.getvalue())

    def test_encode_short(self):
        self.encoder.write(_b('abc'))
        self.encoder.close()
        self.assertEqual(_b('3\r\nabc0\r\n'), self.output.getvalue())

    def test_encode_combines_short(self):
        self.encoder.write(_b('abc'))
        self.encoder.write(_b('def'))
        self.encoder.close()
        self.assertEqual(_b('6\r\nabcdef0\r\n'), self.output.getvalue())

    def test_encode_over_9_is_in_hex(self):
        self.encoder.write(_b('1234567890'))
        self.encoder.close()
        self.assertEqual(_b('A\r\n12345678900\r\n'), self.output.getvalue())

    def test_encode_long_ranges_not_combined(self):
        self.encoder.write(_b('1' * 65536))
        self.encoder.write(_b('2' * 65536))
        self.encoder.close()
        self.assertEqual(_b('10000\r\n' + '1' * 65536 + '10000\r\n' + '2' * 65536 + '0\r\n'), self.output.getvalue())