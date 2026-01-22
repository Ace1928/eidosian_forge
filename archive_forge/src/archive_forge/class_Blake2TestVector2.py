import os
import re
import unittest
import warnings
from binascii import unhexlify, hexlify
from Cryptodome.Util.py3compat import tobytes
from Cryptodome.Util.strxor import strxor_c
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Hash import BLAKE2b, BLAKE2s
class Blake2TestVector2(unittest.TestCase):

    def _load_tests(self, test_vector_file):
        test_vectors = []
        with open(test_vector_file, 'rt') as test_vector_fd:
            for line_number, line in enumerate(test_vector_fd):
                if line.strip() == '' or line.startswith('#'):
                    continue
                res = re.match('digest\\(([0-9]+)\\): ([0-9A-Fa-f]*)', line)
                if not res:
                    raise ValueError('Incorrect test vector format (line %d)' % line_number)
                key_size = int(res.group(1))
                result = unhexlify(tobytes(res.group(2)))
                test_vectors.append((key_size, result))
        return test_vectors

    def setUp(self):
        dir_comps = ('Hash', self.name)
        file_name = 'tv2.txt'
        self.description = '%s tests' % self.name
        try:
            import pycryptodome_test_vectors
        except ImportError:
            warnings.warn('Warning: skipping extended tests for %s' % self.name, UserWarning)
            self.test_vectors = []
            return
        init_dir = os.path.dirname(pycryptodome_test_vectors.__file__)
        full_file_name = os.path.join(os.path.join(init_dir, *dir_comps), file_name)
        self.test_vectors = self._load_tests(full_file_name)

    def runTest(self):
        for key_size, result in self.test_vectors:
            next_data = b''
            for _ in range(100):
                h = self.BLAKE2.new(digest_bytes=self.max_bytes, key=b'A' * key_size)
                h.update(next_data)
                next_data = h.digest() + next_data
            self.assertEqual(h.digest(), result)