import unittest
from binascii import unhexlify
from Cryptodome.SelfTest.loader import load_test_vectors
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Util.py3compat import tobytes, is_string
from Cryptodome.Cipher import AES, DES3, DES
from Cryptodome.Hash import SHAKE128
class BlockChainingTests(unittest.TestCase):
    key_128 = get_tag_random('key_128', 16)
    key_192 = get_tag_random('key_192', 24)
    iv_128 = get_tag_random('iv_128', 16)
    iv_64 = get_tag_random('iv_64', 8)
    data_128 = get_tag_random('data_128', 16)

    def test_loopback_128(self):
        cipher = AES.new(self.key_128, self.aes_mode, self.iv_128)
        pt = get_tag_random('plaintext', 16 * 100)
        ct = cipher.encrypt(pt)
        cipher = AES.new(self.key_128, self.aes_mode, self.iv_128)
        pt2 = cipher.decrypt(ct)
        self.assertEqual(pt, pt2)

    def test_loopback_64(self):
        cipher = DES3.new(self.key_192, self.des3_mode, self.iv_64)
        pt = get_tag_random('plaintext', 8 * 100)
        ct = cipher.encrypt(pt)
        cipher = DES3.new(self.key_192, self.des3_mode, self.iv_64)
        pt2 = cipher.decrypt(ct)
        self.assertEqual(pt, pt2)

    def test_iv(self):
        cipher = AES.new(self.key_128, self.aes_mode)
        iv1 = cipher.iv
        cipher = AES.new(self.key_128, self.aes_mode)
        iv2 = cipher.iv
        self.assertNotEqual(iv1, iv2)
        self.assertEqual(len(iv1), 16)
        cipher = AES.new(self.key_128, self.aes_mode, self.iv_128)
        ct = cipher.encrypt(self.data_128)
        cipher = AES.new(self.key_128, self.aes_mode, iv=self.iv_128)
        self.assertEqual(ct, cipher.encrypt(self.data_128))
        cipher = AES.new(self.key_128, self.aes_mode, IV=self.iv_128)
        self.assertEqual(ct, cipher.encrypt(self.data_128))

    def test_iv_must_be_bytes(self):
        self.assertRaises(TypeError, AES.new, self.key_128, self.aes_mode, iv=u'test1234567890-*')

    def test_only_one_iv(self):
        self.assertRaises(TypeError, AES.new, self.key_128, self.aes_mode, iv=self.iv_128, IV=self.iv_128)

    def test_iv_with_matching_length(self):
        self.assertRaises(ValueError, AES.new, self.key_128, self.aes_mode, b'')
        self.assertRaises(ValueError, AES.new, self.key_128, self.aes_mode, self.iv_128[:15])
        self.assertRaises(ValueError, AES.new, self.key_128, self.aes_mode, self.iv_128 + b'0')

    def test_block_size_128(self):
        cipher = AES.new(self.key_128, self.aes_mode, self.iv_128)
        self.assertEqual(cipher.block_size, AES.block_size)

    def test_block_size_64(self):
        cipher = DES3.new(self.key_192, self.des3_mode, self.iv_64)
        self.assertEqual(cipher.block_size, DES3.block_size)

    def test_unaligned_data_128(self):
        cipher = AES.new(self.key_128, self.aes_mode, self.iv_128)
        for wrong_length in range(1, 16):
            self.assertRaises(ValueError, cipher.encrypt, b'5' * wrong_length)
        cipher = AES.new(self.key_128, self.aes_mode, self.iv_128)
        for wrong_length in range(1, 16):
            self.assertRaises(ValueError, cipher.decrypt, b'5' * wrong_length)

    def test_unaligned_data_64(self):
        cipher = DES3.new(self.key_192, self.des3_mode, self.iv_64)
        for wrong_length in range(1, 8):
            self.assertRaises(ValueError, cipher.encrypt, b'5' * wrong_length)
        cipher = DES3.new(self.key_192, self.des3_mode, self.iv_64)
        for wrong_length in range(1, 8):
            self.assertRaises(ValueError, cipher.decrypt, b'5' * wrong_length)

    def test_IV_iv_attributes(self):
        data = get_tag_random('data', 16 * 100)
        for func in ('encrypt', 'decrypt'):
            cipher = AES.new(self.key_128, self.aes_mode, self.iv_128)
            getattr(cipher, func)(data)
            self.assertEqual(cipher.iv, self.iv_128)
            self.assertEqual(cipher.IV, self.iv_128)

    def test_unknown_parameters(self):
        self.assertRaises(TypeError, AES.new, self.key_128, self.aes_mode, self.iv_128, 7)
        self.assertRaises(TypeError, AES.new, self.key_128, self.aes_mode, iv=self.iv_128, unknown=7)
        AES.new(self.key_128, self.aes_mode, iv=self.iv_128, use_aesni=False)

    def test_null_encryption_decryption(self):
        for func in ('encrypt', 'decrypt'):
            cipher = AES.new(self.key_128, self.aes_mode, self.iv_128)
            result = getattr(cipher, func)(b'')
            self.assertEqual(result, b'')

    def test_either_encrypt_or_decrypt(self):
        cipher = AES.new(self.key_128, self.aes_mode, self.iv_128)
        cipher.encrypt(b'')
        self.assertRaises(TypeError, cipher.decrypt, b'')
        cipher = AES.new(self.key_128, self.aes_mode, self.iv_128)
        cipher.decrypt(b'')
        self.assertRaises(TypeError, cipher.encrypt, b'')

    def test_data_must_be_bytes(self):
        cipher = AES.new(self.key_128, self.aes_mode, self.iv_128)
        self.assertRaises(TypeError, cipher.encrypt, u'test1234567890-*')
        cipher = AES.new(self.key_128, self.aes_mode, self.iv_128)
        self.assertRaises(TypeError, cipher.decrypt, u'test1234567890-*')

    def test_bytearray(self):
        data = b'1' * 128
        data_ba = bytearray(data)
        key_ba = bytearray(self.key_128)
        iv_ba = bytearray(self.iv_128)
        cipher1 = AES.new(self.key_128, self.aes_mode, self.iv_128)
        ref1 = cipher1.encrypt(data)
        cipher2 = AES.new(key_ba, self.aes_mode, iv_ba)
        key_ba[:3] = b'\xff\xff\xff'
        iv_ba[:3] = b'\xff\xff\xff'
        ref2 = cipher2.encrypt(data_ba)
        self.assertEqual(ref1, ref2)
        self.assertEqual(cipher1.iv, cipher2.iv)
        key_ba = bytearray(self.key_128)
        iv_ba = bytearray(self.iv_128)
        cipher3 = AES.new(self.key_128, self.aes_mode, self.iv_128)
        ref3 = cipher3.decrypt(data)
        cipher4 = AES.new(key_ba, self.aes_mode, iv_ba)
        key_ba[:3] = b'\xff\xff\xff'
        iv_ba[:3] = b'\xff\xff\xff'
        ref4 = cipher4.decrypt(data_ba)
        self.assertEqual(ref3, ref4)

    def test_memoryview(self):
        data = b'1' * 128
        data_mv = memoryview(bytearray(data))
        key_mv = memoryview(bytearray(self.key_128))
        iv_mv = memoryview(bytearray(self.iv_128))
        cipher1 = AES.new(self.key_128, self.aes_mode, self.iv_128)
        ref1 = cipher1.encrypt(data)
        cipher2 = AES.new(key_mv, self.aes_mode, iv_mv)
        key_mv[:3] = b'\xff\xff\xff'
        iv_mv[:3] = b'\xff\xff\xff'
        ref2 = cipher2.encrypt(data_mv)
        self.assertEqual(ref1, ref2)
        self.assertEqual(cipher1.iv, cipher2.iv)
        key_mv = memoryview(bytearray(self.key_128))
        iv_mv = memoryview(bytearray(self.iv_128))
        cipher3 = AES.new(self.key_128, self.aes_mode, self.iv_128)
        ref3 = cipher3.decrypt(data)
        cipher4 = AES.new(key_mv, self.aes_mode, iv_mv)
        key_mv[:3] = b'\xff\xff\xff'
        iv_mv[:3] = b'\xff\xff\xff'
        ref4 = cipher4.decrypt(data_mv)
        self.assertEqual(ref3, ref4)

    def test_output_param(self):
        pt = b'5' * 128
        cipher = AES.new(b'4' * 16, self.aes_mode, iv=self.iv_128)
        ct = cipher.encrypt(pt)
        output = bytearray(128)
        cipher = AES.new(b'4' * 16, self.aes_mode, iv=self.iv_128)
        res = cipher.encrypt(pt, output=output)
        self.assertEqual(ct, output)
        self.assertEqual(res, None)
        cipher = AES.new(b'4' * 16, self.aes_mode, iv=self.iv_128)
        res = cipher.decrypt(ct, output=output)
        self.assertEqual(pt, output)
        self.assertEqual(res, None)

    def test_output_param_same_buffer(self):
        pt = b'5' * 128
        cipher = AES.new(b'4' * 16, self.aes_mode, iv=self.iv_128)
        ct = cipher.encrypt(pt)
        pt_ba = bytearray(pt)
        cipher = AES.new(b'4' * 16, self.aes_mode, iv=self.iv_128)
        res = cipher.encrypt(pt_ba, output=pt_ba)
        self.assertEqual(ct, pt_ba)
        self.assertEqual(res, None)
        ct_ba = bytearray(ct)
        cipher = AES.new(b'4' * 16, self.aes_mode, iv=self.iv_128)
        res = cipher.decrypt(ct_ba, output=ct_ba)
        self.assertEqual(pt, ct_ba)
        self.assertEqual(res, None)

    def test_output_param_memoryview(self):
        pt = b'5' * 128
        cipher = AES.new(b'4' * 16, self.aes_mode, iv=self.iv_128)
        ct = cipher.encrypt(pt)
        output = memoryview(bytearray(128))
        cipher = AES.new(b'4' * 16, self.aes_mode, iv=self.iv_128)
        cipher.encrypt(pt, output=output)
        self.assertEqual(ct, output)
        cipher = AES.new(b'4' * 16, self.aes_mode, iv=self.iv_128)
        cipher.decrypt(ct, output=output)
        self.assertEqual(pt, output)

    def test_output_param_neg(self):
        LEN_PT = 128
        pt = b'5' * LEN_PT
        cipher = AES.new(b'4' * 16, self.aes_mode, iv=self.iv_128)
        ct = cipher.encrypt(pt)
        cipher = AES.new(b'4' * 16, self.aes_mode, iv=self.iv_128)
        self.assertRaises(TypeError, cipher.encrypt, pt, output=b'0' * LEN_PT)
        cipher = AES.new(b'4' * 16, self.aes_mode, iv=self.iv_128)
        self.assertRaises(TypeError, cipher.decrypt, ct, output=b'0' * LEN_PT)
        shorter_output = bytearray(LEN_PT - 1)
        cipher = AES.new(b'4' * 16, self.aes_mode, iv=self.iv_128)
        self.assertRaises(ValueError, cipher.encrypt, pt, output=shorter_output)
        cipher = AES.new(b'4' * 16, self.aes_mode, iv=self.iv_128)
        self.assertRaises(ValueError, cipher.decrypt, ct, output=shorter_output)