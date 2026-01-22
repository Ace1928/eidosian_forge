import binascii
import bisect
from ... import tests
from .test_btree_index import compiled_btreeparser_feature
class TestHexAndUnhex(TestBtreeSerializer):

    def assertHexlify(self, as_binary):
        self.assertEqual(binascii.hexlify(as_binary), self.module._py_hexlify(as_binary))

    def assertUnhexlify(self, as_hex):
        ba_unhex = binascii.unhexlify(as_hex)
        mod_unhex = self.module._py_unhexlify(as_hex)
        if ba_unhex != mod_unhex:
            if mod_unhex is None:
                mod_hex = b'<None>'
            else:
                mod_hex = binascii.hexlify(mod_unhex)
            self.fail('_py_unhexlify returned a different answer from binascii:\n    %r\n != %r' % (binascii.hexlify(ba_unhex), mod_hex))

    def assertFailUnhexlify(self, as_hex):
        self.assertIs(None, self.module._py_unhexlify(as_hex))

    def test_to_hex(self):
        raw_bytes = bytes(range(256))
        for i in range(0, 240, 20):
            self.assertHexlify(raw_bytes[i:i + 20])
        self.assertHexlify(raw_bytes[240:] + raw_bytes[0:4])

    def test_from_hex(self):
        self.assertUnhexlify(b'0123456789abcdef0123456789abcdef01234567')
        self.assertUnhexlify(b'123456789abcdef0123456789abcdef012345678')
        self.assertUnhexlify(b'0123456789ABCDEF0123456789ABCDEF01234567')
        self.assertUnhexlify(b'123456789ABCDEF0123456789ABCDEF012345678')
        hex_chars = binascii.hexlify(bytes(range(256)))
        for i in range(0, 480, 40):
            self.assertUnhexlify(hex_chars[i:i + 40])
        self.assertUnhexlify(hex_chars[480:] + hex_chars[0:8])

    def test_from_invalid_hex(self):
        self.assertFailUnhexlify(b'123456789012345678901234567890123456789X')
        self.assertFailUnhexlify(b'12345678901234567890123456789012345678X9')

    def test_bad_argument(self):
        self.assertRaises(ValueError, self.module._py_unhexlify, '1a')
        self.assertRaises(ValueError, self.module._py_unhexlify, b'1b')