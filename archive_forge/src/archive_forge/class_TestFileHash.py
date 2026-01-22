import logging
import os
import tempfile
import time
from hashlib import sha256
from tests.unit import unittest
from boto.compat import BytesIO, six, StringIO
from boto.glacier.utils import minimum_part_size, chunk_hashes, tree_hash, \
class TestFileHash(unittest.TestCase):

    def _gen_data(self):
        return os.urandom(5000) + b'\xc2\x00'

    def test_compute_hash_tempfile(self):
        if six.PY2:
            mode = 'w+'
        else:
            mode = 'wb+'
        with tempfile.TemporaryFile(mode=mode) as f:
            f.write(self._gen_data())
            f.seek(0)
            compute_hashes_from_fileobj(f, chunk_size=512)

    @unittest.skipUnless(six.PY3, 'Python 3 requires reading binary!')
    def test_compute_hash_tempfile_py3(self):
        with tempfile.TemporaryFile(mode='w+') as f:
            with self.assertRaises(ValueError):
                compute_hashes_from_fileobj(f, chunk_size=512)
        f = StringIO('test data' * 500)
        compute_hashes_from_fileobj(f, chunk_size=512)

    @unittest.skipUnless(six.PY2, 'Python 3 requires reading binary!')
    def test_compute_hash_stringio(self):
        f = StringIO(self._gen_data())
        compute_hashes_from_fileobj(f, chunk_size=512)

    def test_compute_hash_bytesio(self):
        f = BytesIO(self._gen_data())
        compute_hashes_from_fileobj(f, chunk_size=512)