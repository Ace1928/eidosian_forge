import unittest
from binascii import unhexlify
from Cryptodome.SelfTest.loader import load_test_vectors
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Util.py3compat import tobytes, is_string
from Cryptodome.Cipher import AES, DES3, DES
from Cryptodome.Hash import SHAKE128
def _do_tdes_test(self, file_name):
    test_vectors = load_test_vectors(('Cipher', 'TDES'), file_name, 'TDES CBC KAT', {'count': lambda x: int(x)})
    if test_vectors is None:
        return
    direction = None
    for tv in test_vectors:
        if is_string(tv):
            direction = tv
            continue
        self.description = tv.desc
        if hasattr(tv, 'keys'):
            cipher = DES.new(tv.keys, self.des_mode, tv.iv)
        else:
            if tv.key1 != tv.key3:
                key = tv.key1 + tv.key2 + tv.key3
            else:
                key = tv.key1 + tv.key2
            cipher = DES3.new(key, self.des3_mode, tv.iv)
        if direction == '[ENCRYPT]':
            self.assertEqual(cipher.encrypt(tv.plaintext), tv.ciphertext)
        elif direction == '[DECRYPT]':
            self.assertEqual(cipher.decrypt(tv.ciphertext), tv.plaintext)
        else:
            assert False