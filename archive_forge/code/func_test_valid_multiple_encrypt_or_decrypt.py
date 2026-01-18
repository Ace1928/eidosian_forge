import unittest
from binascii import unhexlify
from Cryptodome.Util.py3compat import b, tobytes, bchr
from Cryptodome.Util.number import long_to_bytes
from Cryptodome.SelfTest.loader import load_test_vectors
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Cipher import AES
from Cryptodome.Hash import SHAKE128
def test_valid_multiple_encrypt_or_decrypt(self):
    for method_name in ('encrypt', 'decrypt'):
        for auth_data in (None, b('333'), self.data, self.data + b('3')):
            cipher = AES.new(self.key_128, AES.MODE_OCB, nonce=self.nonce_96)
            if auth_data is not None:
                cipher.update(auth_data)
            method = getattr(cipher, method_name)
            method(self.data)
            method(self.data)
            method(self.data)
            method(self.data)
            method()