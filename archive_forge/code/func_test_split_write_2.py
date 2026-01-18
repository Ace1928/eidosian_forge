from hashlib import sha256
import itertools
from boto.compat import StringIO
from tests.unit import unittest
from mock import (
from nose.tools import assert_equal
from boto.glacier.layer1 import Layer1
from boto.glacier.vault import Vault
from boto.glacier.writer import Writer, resume_file_upload
from boto.glacier.utils import bytes_to_hex, chunk_hashes, tree_hash
def test_split_write_2(self):
    self.check_write([b'12', b'34'])