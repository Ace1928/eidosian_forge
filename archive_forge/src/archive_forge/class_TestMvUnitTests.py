from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
from gslib.cs_api_map import ApiSelector
from gslib.tests.test_cp import TestCpMvPOSIXBucketToLocalErrors
from gslib.tests.test_cp import TestCpMvPOSIXBucketToLocalNoErrors
from gslib.tests.test_cp import TestCpMvPOSIXLocalToBucketNoErrors
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import SequentialAndParallelTransfer
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import SetEnvironmentForTest
from gslib.tests.util import unittest
from gslib.utils.boto_util import UsingCrcmodExtension
from gslib.utils.retry_util import Retry
from gslib.utils.system_util import IS_WINDOWS
from gslib.utils import shim_util
class TestMvUnitTests(testcase.GsUtilUnitTestCase):
    """Unit tests for mv command."""

    def test_move_bucket_objects_with_duplicate_names_inter_bucket(self):
        """Tests moving multiple top-level items between buckets."""
        bucket1_uri = self.CreateBucket()
        self.CreateObject(bucket_uri=bucket1_uri, object_name='dir1/file.txt', contents=b'data')
        self.CreateObject(bucket_uri=bucket1_uri, object_name='dir2/file.txt', contents=b'data')
        bucket2_uri = self.CreateBucket()
        self.RunCommand('mv', [suri(bucket1_uri, '*'), suri(bucket2_uri)])
        actual = set((str(u) for u in self._test_wildcard_iterator(suri(bucket2_uri, '**')).IterAll(expand_top_level_buckets=True)))
        expected = set([suri(bucket2_uri, 'dir1', 'file.txt'), suri(bucket2_uri, 'dir2', 'file.txt')])
        self.assertEqual(actual, expected)

    def test_move_bucket_objects_with_duplicate_names_to_bucket_subdir(self):
        """Tests moving multiple top-level items between buckets."""
        bucket1_uri = self.CreateBucket()
        self.CreateObject(bucket_uri=bucket1_uri, object_name='dir1/file.txt', contents=b'data')
        self.CreateObject(bucket_uri=bucket1_uri, object_name='dir2/file.txt', contents=b'data')
        bucket2_uri = self.CreateBucket()
        self.RunCommand('mv', [suri(bucket1_uri, '*'), suri(bucket2_uri, 'dir')])
        actual = set((str(u) for u in self._test_wildcard_iterator(suri(bucket2_uri, '**')).IterAll(expand_top_level_buckets=True)))
        expected = set([suri(bucket2_uri, 'dir', 'dir1', 'file.txt'), suri(bucket2_uri, 'dir', 'dir2', 'file.txt')])
        self.assertEqual(actual, expected)

    def test_move_dirs_with_duplicate_file_names_to_bucket(self):
        """Tests moving multiple top-level items to a bucket."""
        bucket_uri = self.CreateBucket()
        dir_path = self.CreateTempDir(test_files=[('dir1', 'file.txt'), ('dir2', 'file.txt')])
        self.RunCommand('mv', [dir_path + '/*', suri(bucket_uri)])
        actual = set((str(u) for u in self._test_wildcard_iterator(suri(bucket_uri, '**')).IterAll(expand_top_level_buckets=True)))
        expected = set([suri(bucket_uri, 'dir1', 'file.txt'), suri(bucket_uri, 'dir2', 'file.txt')])
        self.assertEqual(actual, expected)

    def test_shim_translates_flags(self):
        bucket_uri = self.CreateBucket()
        fpath = self.CreateTempFile(contents=b'abcd')
        with SetBotoConfigForTest([('GSUtil', 'use_gcloud_storage', 'True'), ('GSUtil', 'hidden_shim_mode', 'dry_run')]):
            with SetEnvironmentForTest({'CLOUDSDK_CORE_PASS_CREDENTIALS_TO_GSUTIL': 'True', 'CLOUDSDK_ROOT_DIR': 'fake_dir'}):
                mock_log_handler = self.RunCommand('mv', ['-a', 'public-read', fpath, suri(bucket_uri)], return_log_handler=True)
                info_lines = '\n'.join(mock_log_handler.messages['info'])
                self.assertIn('Gcloud Storage Command: {} storage mv --predefined-acl publicRead {} {}'.format(shim_util._get_gcloud_binary_path('fake_dir'), fpath, suri(bucket_uri)), info_lines)