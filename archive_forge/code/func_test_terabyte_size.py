import logging
import os
import tempfile
import time
from hashlib import sha256
from tests.unit import unittest
from boto.compat import BytesIO, six, StringIO
from boto.glacier.utils import minimum_part_size, chunk_hashes, tree_hash, \
def test_terabyte_size(self):
    self.assertEqual(minimum_part_size(4 * 1024 * 1024 * 1024 * 1024), 512 * 1024 * 1024)