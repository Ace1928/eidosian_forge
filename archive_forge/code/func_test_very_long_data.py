import unittest
from binascii import hexlify, unhexlify
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Util.py3compat import tobytes, bchr
from Cryptodome.Cipher import AES, DES3
from Cryptodome.Hash import SHAKE128, SHA256
from Cryptodome.Util import Counter
def test_very_long_data(self):
    cipher = AES.new(b'A' * 32, AES.MODE_CTR, nonce=b'')
    ct = cipher.encrypt(b'B' * 1000000)
    digest = SHA256.new(ct).hexdigest()
    self.assertEqual(digest, '96204fc470476561a3a8f3b6fe6d24be85c87510b638142d1d0fb90989f8a6a6')