import os
import re
import unittest
import warnings
from binascii import unhexlify, hexlify
from Cryptodome.Util.py3compat import tobytes
from Cryptodome.Util.strxor import strxor_c
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Hash import BLAKE2b, BLAKE2s
class Blake2TestVector1(unittest.TestCase):

    def _load_tests(self, test_vector_file):
        test_vectors = []
        with open(test_vector_file, 'rt') as test_vector_fd:
            for line_number, line in enumerate(test_vector_fd):
                if line.strip() == '' or line.startswith('#'):
                    continue
                res = re.match('digest: ([0-9A-Fa-f]*)', line)
                if not res:
                    raise ValueError('Incorrect test vector format (line %d)' % line_number)
                test_vectors.append(unhexlify(tobytes(res.group(1))))
        return test_vectors

    def setUp(self):
        dir_comps = ('Hash', self.name)
        file_name = 'tv1.txt'
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
        for tv in self.test_vectors:
            digest_bytes = len(tv)
            next_data = b''
            for _ in range(100):
                h = self.BLAKE2.new(digest_bytes=digest_bytes)
                h.update(next_data)
                next_data = h.digest() + next_data
            self.assertEqual(h.digest(), tv)