import unittest
from binascii import unhexlify
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.SelfTest.loader import load_test_vectors_wycheproof
from Cryptodome.Util.py3compat import tobytes
from Cryptodome.Cipher import ChaCha20_Poly1305
from Cryptodome.Hash import SHAKE128
from Cryptodome.Util.strxor import strxor
class XChaCha20Poly1305Tests(unittest.TestCase):

    def test_nonce(self):
        cipher = ChaCha20_Poly1305.new(key=b'Y' * 32, nonce=b'H' * 24)
        self.assertEqual(len(cipher.nonce), 24)
        self.assertEqual(cipher.nonce, b'H' * 24)

    def test_encrypt(self):
        pt = b'\n                4c616469657320616e642047656e746c656d656e206f662074686520636c6173\n                73206f66202739393a204966204920636f756c64206f6666657220796f75206f\n                6e6c79206f6e652074697020666f7220746865206675747572652c2073756e73\n                637265656e20776f756c642062652069742e'
        pt = unhexlify(pt.replace(b'\n', b'').replace(b' ', b''))
        aad = unhexlify(b'50515253c0c1c2c3c4c5c6c7')
        key = unhexlify(b'808182838485868788898a8b8c8d8e8f909192939495969798999a9b9c9d9e9f')
        iv = unhexlify(b'404142434445464748494a4b4c4d4e4f5051525354555657')
        ct = b'\n                bd6d179d3e83d43b9576579493c0e939572a1700252bfaccbed2902c21396cbb\n                731c7f1b0b4aa6440bf3a82f4eda7e39ae64c6708c54c216cb96b72e1213b452\n                2f8c9ba40db5d945b11b69b982c1bb9e3f3fac2bc369488f76b2383565d3fff9\n                21f9664c97637da9768812f615c68b13b52e'
        ct = unhexlify(ct.replace(b'\n', b'').replace(b' ', b''))
        tag = unhexlify(b'c0875924c1c7987947deafd8780acf49')
        cipher = ChaCha20_Poly1305.new(key=key, nonce=iv)
        cipher.update(aad)
        ct_test, tag_test = cipher.encrypt_and_digest(pt)
        self.assertEqual(ct, ct_test)
        self.assertEqual(tag, tag_test)
        cipher = ChaCha20_Poly1305.new(key=key, nonce=iv)
        cipher.update(aad)
        cipher.decrypt_and_verify(ct, tag)