import binascii
import bisect
from ... import tests
from .test_btree_index import compiled_btreeparser_feature
class Test_KeyToSha1(TestBtreeSerializer):

    def assertKeyToSha1(self, expected, key):
        if expected is None:
            expected_bin = None
        else:
            expected_bin = binascii.unhexlify(expected)
        actual_sha1 = self.module._py_key_to_sha1(key)
        if expected_bin != actual_sha1:
            actual_hex_sha1 = None
            if actual_sha1 is not None:
                actual_hex_sha1 = binascii.hexlify(actual_sha1)
            self.fail('_key_to_sha1 returned:\n    %s\n != %s' % (actual_sha1, expected))

    def test_simple(self):
        self.assertKeyToSha1(_hex_form, (b'sha1:' + _hex_form,))

    def test_invalid_not_tuple(self):
        self.assertKeyToSha1(None, _hex_form)
        self.assertKeyToSha1(None, b'sha1:' + _hex_form)

    def test_invalid_empty(self):
        self.assertKeyToSha1(None, ())

    def test_invalid_not_string(self):
        self.assertKeyToSha1(None, (None,))
        self.assertKeyToSha1(None, (list(_hex_form),))

    def test_invalid_not_sha1(self):
        self.assertKeyToSha1(None, (_hex_form,))
        self.assertKeyToSha1(None, (b'sha2:' + _hex_form,))

    def test_invalid_not_hex(self):
        self.assertKeyToSha1(None, (b'sha1:abcdefghijklmnopqrstuvwxyz12345678901234',))