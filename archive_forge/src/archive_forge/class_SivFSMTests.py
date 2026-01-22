import json
import unittest
from binascii import unhexlify
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.SelfTest.loader import load_test_vectors_wycheproof
from Cryptodome.Util.py3compat import tobytes, bchr
from Cryptodome.Cipher import AES
from Cryptodome.Hash import SHAKE128
from Cryptodome.Util.strxor import strxor
class SivFSMTests(unittest.TestCase):
    key_256 = get_tag_random('key_256', 32)
    nonce_96 = get_tag_random('nonce_96', 12)
    data = get_tag_random('data', 128)

    def test_invalid_init_encrypt(self):
        cipher = AES.new(self.key_256, AES.MODE_SIV, nonce=self.nonce_96)
        self.assertRaises(TypeError, cipher.encrypt, b'xxx')

    def test_invalid_init_decrypt(self):
        cipher = AES.new(self.key_256, AES.MODE_SIV, nonce=self.nonce_96)
        self.assertRaises(TypeError, cipher.decrypt, b'xxx')

    def test_valid_init_update_digest_verify(self):
        cipher = AES.new(self.key_256, AES.MODE_SIV, nonce=self.nonce_96)
        cipher.update(self.data)
        mac = cipher.digest()
        cipher = AES.new(self.key_256, AES.MODE_SIV, nonce=self.nonce_96)
        cipher.update(self.data)
        cipher.verify(mac)

    def test_valid_init_digest(self):
        cipher = AES.new(self.key_256, AES.MODE_SIV, nonce=self.nonce_96)
        cipher.digest()

    def test_valid_init_verify(self):
        cipher = AES.new(self.key_256, AES.MODE_SIV, nonce=self.nonce_96)
        mac = cipher.digest()
        cipher = AES.new(self.key_256, AES.MODE_SIV, nonce=self.nonce_96)
        cipher.verify(mac)

    def test_valid_multiple_digest_or_verify(self):
        cipher = AES.new(self.key_256, AES.MODE_SIV, nonce=self.nonce_96)
        cipher.update(self.data)
        first_mac = cipher.digest()
        for x in range(4):
            self.assertEqual(first_mac, cipher.digest())
        cipher = AES.new(self.key_256, AES.MODE_SIV, nonce=self.nonce_96)
        cipher.update(self.data)
        for x in range(5):
            cipher.verify(first_mac)

    def test_valid_encrypt_and_digest_decrypt_and_verify(self):
        cipher = AES.new(self.key_256, AES.MODE_SIV, nonce=self.nonce_96)
        cipher.update(self.data)
        ct, mac = cipher.encrypt_and_digest(self.data)
        cipher = AES.new(self.key_256, AES.MODE_SIV, nonce=self.nonce_96)
        cipher.update(self.data)
        pt = cipher.decrypt_and_verify(ct, mac)
        self.assertEqual(self.data, pt)

    def test_invalid_multiple_encrypt_and_digest(self):
        cipher = AES.new(self.key_256, AES.MODE_SIV, nonce=self.nonce_96)
        ct, tag = cipher.encrypt_and_digest(self.data)
        self.assertRaises(TypeError, cipher.encrypt_and_digest, b'')

    def test_invalid_multiple_decrypt_and_verify(self):
        cipher = AES.new(self.key_256, AES.MODE_SIV, nonce=self.nonce_96)
        ct, tag = cipher.encrypt_and_digest(self.data)
        cipher = AES.new(self.key_256, AES.MODE_SIV, nonce=self.nonce_96)
        cipher.decrypt_and_verify(ct, tag)
        self.assertRaises(TypeError, cipher.decrypt_and_verify, ct, tag)