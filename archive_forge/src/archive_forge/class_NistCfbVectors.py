import unittest
from binascii import unhexlify
from Cryptodome.SelfTest.loader import load_test_vectors
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Util.py3compat import tobytes, is_string
from Cryptodome.Cipher import AES, DES3, DES
from Cryptodome.Hash import SHAKE128
from Cryptodome.SelfTest.Cipher.test_CBC import BlockChainingTests
class NistCfbVectors(unittest.TestCase):

    def _do_kat_aes_test(self, file_name, segment_size):
        test_vectors = load_test_vectors(('Cipher', 'AES'), file_name, 'AES CFB%d KAT' % segment_size, {'count': lambda x: int(x)})
        if test_vectors is None:
            return
        direction = None
        for tv in test_vectors:
            if is_string(tv):
                direction = tv
                continue
            self.description = tv.desc
            cipher = AES.new(tv.key, AES.MODE_CFB, tv.iv, segment_size=segment_size)
            if direction == '[ENCRYPT]':
                self.assertEqual(cipher.encrypt(tv.plaintext), tv.ciphertext)
            elif direction == '[DECRYPT]':
                self.assertEqual(cipher.decrypt(tv.ciphertext), tv.plaintext)
            else:
                assert False

    def _do_mct_aes_test(self, file_name, segment_size):
        test_vectors = load_test_vectors(('Cipher', 'AES'), file_name, 'AES CFB%d Montecarlo' % segment_size, {'count': lambda x: int(x)})
        if test_vectors is None:
            return
        assert segment_size in (8, 128)
        direction = None
        for tv in test_vectors:
            if is_string(tv):
                direction = tv
                continue
            self.description = tv.desc
            cipher = AES.new(tv.key, AES.MODE_CFB, tv.iv, segment_size=segment_size)

            def get_input(input_text, output_seq, j):
                if segment_size == 128:
                    if j >= 2:
                        return output_seq[-2]
                    return [input_text, tv.iv][j]
                if j == 0:
                    return input_text
                elif j <= 16:
                    return tv.iv[j - 1:j]
                return output_seq[j - 17]
            if direction == '[ENCRYPT]':
                cts = []
                for j in range(1000):
                    plaintext = get_input(tv.plaintext, cts, j)
                    cts.append(cipher.encrypt(plaintext))
                self.assertEqual(cts[-1], tv.ciphertext)
            elif direction == '[DECRYPT]':
                pts = []
                for j in range(1000):
                    ciphertext = get_input(tv.ciphertext, pts, j)
                    pts.append(cipher.decrypt(ciphertext))
                self.assertEqual(pts[-1], tv.plaintext)
            else:
                assert False

    def _do_tdes_test(self, file_name, segment_size):
        test_vectors = load_test_vectors(('Cipher', 'TDES'), file_name, 'TDES CFB%d KAT' % segment_size, {'count': lambda x: int(x)})
        if test_vectors is None:
            return
        direction = None
        for tv in test_vectors:
            if is_string(tv):
                direction = tv
                continue
            self.description = tv.desc
            if hasattr(tv, 'keys'):
                cipher = DES.new(tv.keys, DES.MODE_CFB, tv.iv, segment_size=segment_size)
            else:
                if tv.key1 != tv.key3:
                    key = tv.key1 + tv.key2 + tv.key3
                else:
                    key = tv.key1 + tv.key2
                cipher = DES3.new(key, DES3.MODE_CFB, tv.iv, segment_size=segment_size)
            if direction == '[ENCRYPT]':
                self.assertEqual(cipher.encrypt(tv.plaintext), tv.ciphertext)
            elif direction == '[DECRYPT]':
                self.assertEqual(cipher.decrypt(tv.ciphertext), tv.plaintext)
            else:
                assert False