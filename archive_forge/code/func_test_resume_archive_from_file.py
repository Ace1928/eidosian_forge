from tests.unit import unittest
from mock import call, Mock, patch, sentinel
import codecs
from boto.glacier.layer1 import Layer1
from boto.glacier.layer2 import Layer2
import boto.glacier.vault
from boto.glacier.vault import Vault
from boto.glacier.vault import Job
from datetime import datetime, tzinfo, timedelta
@patch('boto.glacier.vault.resume_file_upload')
def test_resume_archive_from_file(self, mock_resume_file_upload):
    part_size = 4
    mock_list_parts = Mock()
    mock_list_parts.return_value = {'PartSizeInBytes': part_size, 'Parts': [{'RangeInBytes': '0-3', 'SHA256TreeHash': '12'}, {'RangeInBytes': '4-6', 'SHA256TreeHash': '34'}]}
    self.vault.list_all_parts = mock_list_parts
    self.vault.resume_archive_from_file(sentinel.upload_id, file_obj=sentinel.file_obj)
    mock_resume_file_upload.assert_called_once_with(self.vault, sentinel.upload_id, part_size, sentinel.file_obj, {0: codecs.decode('12', 'hex_codec'), 1: codecs.decode('34', 'hex_codec')})