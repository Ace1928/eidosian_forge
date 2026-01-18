import unittest
from binascii import unhexlify
from Cryptodome.Util.py3compat import b, tobytes, bchr
from Cryptodome.Util.number import long_to_bytes
from Cryptodome.SelfTest.loader import load_test_vectors
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Cipher import AES
from Cryptodome.Hash import SHAKE128
def test_valid_full_path(self):
    cipher = AES.new(self.key_128, AES.MODE_OCB, nonce=self.nonce_96)
    cipher.update(self.data)
    ct = cipher.encrypt(self.data)
    ct += cipher.encrypt()
    mac = cipher.digest()
    cipher = AES.new(self.key_128, AES.MODE_OCB, nonce=self.nonce_96)
    cipher.update(self.data)
    cipher.decrypt(ct)
    cipher.decrypt()
    cipher.verify(mac)
    cipher = AES.new(self.key_128, AES.MODE_OCB, nonce=self.nonce_96)
    cipher.update(self.data)
    ct1 = cipher.encrypt(self.data[:2])
    ct2, mac = cipher.encrypt_and_digest(self.data[2:])
    cipher = AES.new(self.key_128, AES.MODE_OCB, nonce=self.nonce_96)
    cipher.update(self.data)
    cipher.decrypt(ct1)
    cipher.decrypt_and_verify(ct2, mac)