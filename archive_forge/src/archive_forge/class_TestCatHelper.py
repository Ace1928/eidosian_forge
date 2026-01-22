from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
import sys
from gslib.cs_api_map import ApiSelector
from gslib.exception import NO_URLS_MATCHED_TARGET
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.util import GenerationFromURI as urigen
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import RUN_S3_TESTS
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import SetEnvironmentForTest
from gslib.tests.util import TEST_ENCRYPTION_KEY1
from gslib.tests.util import unittest
from gslib.utils import cat_helper
from gslib.utils import shim_util
from unittest import mock
class TestCatHelper(testcase.GsUtilUnitTestCase):
    """Unit tests for cat helper."""

    def test_cat_helper_runs_flush(self):
        cat_command_mock = mock.Mock()
        cat_helper_mock = cat_helper.CatHelper(command_obj=cat_command_mock)
        object_contents = '0123456789'
        bucket_uri = self.CreateBucket(bucket_name='bucket', provider=self.default_provider)
        obj = self.CreateObject(bucket_uri=bucket_uri, object_name='foo1', contents=object_contents)
        obj1 = self.CreateObject(bucket_uri=bucket_uri, object_name='foo2', contents=object_contents)
        cat_command_mock.WildcardIterator.return_value = self._test_wildcard_iterator('gs://bucket/foo*')
        stdout_mock = mock.mock_open()()
        write_flush_collector_mock = mock.Mock()
        cat_command_mock.gsutil_api.GetObjectMedia = write_flush_collector_mock
        stdout_mock.flush = write_flush_collector_mock
        cat_helper_mock.CatUrlStrings(url_strings=['url'], cat_out_fd=stdout_mock)
        mock_part_one = [mock.call('bucket', 'foo1', stdout_mock, compressed_encoding=None, start_byte=0, end_byte=None, object_size=10, generation=None, decryption_tuple=None, provider='gs'), mock.call()]
        mock_part_two = [mock.call('bucket', 'foo2', stdout_mock, compressed_encoding=None, start_byte=0, end_byte=None, object_size=10, generation=None, decryption_tuple=None, provider='gs'), mock.call()]
        self.assertIn(write_flush_collector_mock.call_args_list[0:2], [mock_part_one, mock_part_two])
        self.assertIn(write_flush_collector_mock.call_args_list[2:4], [mock_part_one, mock_part_two])