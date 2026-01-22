from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import re
from unittest import skipIf
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.testcase.integration_testcase import SkipForXML
from gslib.tests.util import ObjectToURI as suri
@SkipForS3('S3 does not support storage class at bucket level.')
class TestDefStorageClass(testcase.GsUtilIntegrationTestCase):
    """Integration tests for defstorageclass command."""
    _set_dsc_cmd = ['defstorageclass', 'set']
    _get_dsc_cmd = ['defstorageclass', 'get']

    def test_set_and_get_for_one_bucket(self):
        bucket_uri = self.CreateBucket()
        new_storage_class = 'nearline'
        stderr = self.RunGsUtil(self._set_dsc_cmd + [new_storage_class, suri(bucket_uri)], return_stderr=True)
        if not self._use_gcloud_storage:
            self.assertRegexpMatchesWithFlags(stderr, 'Setting default storage class to "%s" for bucket %s' % (new_storage_class, suri(bucket_uri)), flags=re.IGNORECASE)
        stdout = self.RunGsUtil(self._get_dsc_cmd + [suri(bucket_uri)], return_stdout=True)
        self.assertRegexpMatchesWithFlags(stdout, '%s:\\s+%s' % (suri(bucket_uri), new_storage_class), flags=re.IGNORECASE)

    def test_set_and_get_for_multiple_buckets(self):
        bucket1_uri = self.CreateBucket()
        bucket2_uri = self.CreateBucket()
        new_storage_class = 'nearline'
        stderr = self.RunGsUtil(self._set_dsc_cmd + [new_storage_class, suri(bucket1_uri), suri(bucket2_uri)], return_stderr=True)
        for bucket_uri in (suri(bucket1_uri), suri(bucket2_uri)):
            if not self._use_gcloud_storage:
                self.assertRegexpMatchesWithFlags(stderr, 'Setting default storage class to "%s" for bucket %s' % (new_storage_class, bucket_uri), flags=re.IGNORECASE)
        stdout = self.RunGsUtil(self._get_dsc_cmd + [suri(bucket1_uri), suri(bucket2_uri)], return_stdout=True)
        for bucket_uri in (suri(bucket1_uri), suri(bucket2_uri)):
            self.assertRegexpMatchesWithFlags(stdout, '%s:\\s+%s' % (bucket_uri, new_storage_class), flags=re.IGNORECASE)

    def test_set_invalid_storage_class_fails(self):
        bucket_uri = self.CreateBucket()
        stderr = self.RunGsUtil(self._set_dsc_cmd + ['invalidclass', suri(bucket_uri)], return_stderr=True, expected_status=1)
        if self._use_gcloud_storage:
            self.assertIn('Invalid storage class', stderr)
        else:
            self.assertIn('BadRequestException: 400', stderr)

    def test_too_few_arguments_fails(self):
        stderr = self.RunGsUtil(self._set_dsc_cmd, return_stderr=True, expected_status=1)
        self.assertIn('command requires at least', stderr)
        if self._use_gcloud_storage:
            expected_status = 2
            expected_error_string = 'argument URL [URL ...]: Must be specified'
        else:
            expected_status = 1
            expected_error_string = 'command requires at least'
        stderr = self.RunGsUtil(self._set_dsc_cmd + ['std'], return_stderr=True, expected_status=expected_status)
        self.assertIn(expected_error_string, stderr)
        stderr = self.RunGsUtil(self._get_dsc_cmd, return_stderr=True, expected_status=1)
        self.assertIn('command requires at least', stderr)

    def test_helpful_failure_with_s3_urls(self):
        s3_bucket_url = 's3://somebucket'
        failure_msg = 'does not support the URL "%s"' % s3_bucket_url
        stderr = self.RunGsUtil(self._get_dsc_cmd + [s3_bucket_url], return_stderr=True, expected_status=1)
        self.assertIn(failure_msg, stderr)
        stderr = self.RunGsUtil(self._set_dsc_cmd + ['ClassFoo', s3_bucket_url], return_stderr=True, expected_status=1)
        self.assertIn(failure_msg, stderr)