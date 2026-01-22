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
class TestResume(unittest.TestCase):

    def setUp(self):
        super(TestResume, self).setUp()
        self.vault = create_mock_vault()
        self.chunk_size = 2
        self.part_size = 4

    def check_no_resume(self, data, resume_set=set()):
        fobj = StringIO(data.decode('utf-8'))
        part_hash_map = {}
        for part_index in resume_set:
            start = self.part_size * part_index
            end = start + self.part_size
            part_data = data[start:end]
            part_hash_map[part_index] = tree_hash(chunk_hashes(part_data, self.chunk_size))
        resume_file_upload(self.vault, sentinel.upload_id, self.part_size, fobj, part_hash_map, self.chunk_size)
        upload_part_calls, data_tree_hashes = calculate_mock_vault_calls(data, self.part_size, self.chunk_size)
        resume_upload_part_calls = [call for part_index, call in enumerate(upload_part_calls) if part_index not in resume_set]
        check_mock_vault_calls(self.vault, resume_upload_part_calls, data_tree_hashes, len(data))

    def test_one_part_no_resume(self):
        self.check_no_resume(b'1234')

    def test_two_parts_no_resume(self):
        self.check_no_resume(b'12345678')

    def test_one_part_resume(self):
        self.check_no_resume(b'1234', resume_set=set([0]))

    def test_two_parts_one_resume(self):
        self.check_no_resume(b'12345678', resume_set=set([1]))

    def test_returns_archive_id(self):
        archive_id = resume_file_upload(self.vault, sentinel.upload_id, self.part_size, StringIO('1'), {}, self.chunk_size)
        self.assertEquals(sentinel.archive_id, archive_id)