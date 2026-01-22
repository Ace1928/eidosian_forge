import sys
from ... import tests
from ...tests import features
from ...tests.scenarios import load_tests_apply_scenarios
from .. import _groupcompress_py
class TestBase128Int(tests.TestCase):
    scenarios = module_scenarios()
    _gc_module = None

    def assertEqualEncode(self, bytes, val):
        self.assertEqual(bytes, self._gc_module.encode_base128_int(val))

    def assertEqualDecode(self, val, num_decode, bytes):
        self.assertEqual((val, num_decode), self._gc_module.decode_base128_int(bytes))

    def test_encode(self):
        self.assertEqualEncode(b'\x01', 1)
        self.assertEqualEncode(b'\x02', 2)
        self.assertEqualEncode(b'\x7f', 127)
        self.assertEqualEncode(b'\x80\x01', 128)
        self.assertEqualEncode(b'\xff\x01', 255)
        self.assertEqualEncode(b'\x80\x02', 256)
        self.assertEqualEncode(b'\xff\xff\xff\xff\x0f', 4294967295)

    def test_decode(self):
        self.assertEqualDecode(1, 1, b'\x01')
        self.assertEqualDecode(2, 1, b'\x02')
        self.assertEqualDecode(127, 1, b'\x7f')
        self.assertEqualDecode(128, 2, b'\x80\x01')
        self.assertEqualDecode(255, 2, b'\xff\x01')
        self.assertEqualDecode(256, 2, b'\x80\x02')
        self.assertEqualDecode(4294967295, 5, b'\xff\xff\xff\xff\x0f')

    def test_decode_with_trailing_bytes(self):
        self.assertEqualDecode(1, 1, b'\x01abcdef')
        self.assertEqualDecode(127, 1, b'\x7f\x01')
        self.assertEqualDecode(128, 2, b'\x80\x01abcdef')
        self.assertEqualDecode(255, 2, b'\xff\x01\xff')