import unittest
from binascii import unhexlify
from Cryptodome.SelfTest.loader import load_test_vectors
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Util.py3compat import tobytes, is_string
from Cryptodome.Cipher import AES, DES3, DES
from Cryptodome.Hash import SHAKE128
def _do_mct_aes_test(self, file_name):
    test_vectors = load_test_vectors(('Cipher', 'AES'), file_name, 'AES CBC Montecarlo', {'count': lambda x: int(x)})
    if test_vectors is None:
        return
    direction = None
    for tv in test_vectors:
        if is_string(tv):
            direction = tv
            continue
        self.description = tv.desc
        cipher = AES.new(tv.key, self.aes_mode, tv.iv)
        if direction == '[ENCRYPT]':
            cts = [tv.iv]
            for count in range(1000):
                cts.append(cipher.encrypt(tv.plaintext))
                tv.plaintext = cts[-2]
            self.assertEqual(cts[-1], tv.ciphertext)
        elif direction == '[DECRYPT]':
            pts = [tv.iv]
            for count in range(1000):
                pts.append(cipher.decrypt(tv.ciphertext))
                tv.ciphertext = pts[-2]
            self.assertEqual(pts[-1], tv.plaintext)
        else:
            assert False