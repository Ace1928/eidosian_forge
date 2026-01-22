from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from datetime import datetime
import os
import posixpath
import re
import stat
import subprocess
import sys
import time
import gslib
from gslib.commands import ls
from gslib.cs_api_map import ApiSelector
from gslib.project_id import PopulateProjectId
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForGS
from gslib.tests.testcase.integration_testcase import SkipForJSON
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.testcase.integration_testcase import SkipForXML
from gslib.tests.util import CaptureStdout
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import RUN_S3_TESTS
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import SetEnvironmentForTest
from gslib.tests.util import TEST_ENCRYPTION_CONTENT1
from gslib.tests.util import TEST_ENCRYPTION_CONTENT1_CRC32C
from gslib.tests.util import TEST_ENCRYPTION_CONTENT1_MD5
from gslib.tests.util import TEST_ENCRYPTION_CONTENT2
from gslib.tests.util import TEST_ENCRYPTION_CONTENT2_CRC32C
from gslib.tests.util import TEST_ENCRYPTION_CONTENT2_MD5
from gslib.tests.util import TEST_ENCRYPTION_CONTENT3
from gslib.tests.util import TEST_ENCRYPTION_CONTENT3_CRC32C
from gslib.tests.util import TEST_ENCRYPTION_CONTENT3_MD5
from gslib.tests.util import TEST_ENCRYPTION_CONTENT4
from gslib.tests.util import TEST_ENCRYPTION_CONTENT4_CRC32C
from gslib.tests.util import TEST_ENCRYPTION_CONTENT4_MD5
from gslib.tests.util import TEST_ENCRYPTION_CONTENT5
from gslib.tests.util import TEST_ENCRYPTION_CONTENT5_CRC32C
from gslib.tests.util import TEST_ENCRYPTION_CONTENT5_MD5
from gslib.tests.util import TEST_ENCRYPTION_KEY1
from gslib.tests.util import TEST_ENCRYPTION_KEY1_SHA256_B64
from gslib.tests.util import TEST_ENCRYPTION_KEY2
from gslib.tests.util import TEST_ENCRYPTION_KEY2_SHA256_B64
from gslib.tests.util import TEST_ENCRYPTION_KEY3
from gslib.tests.util import TEST_ENCRYPTION_KEY3_SHA256_B64
from gslib.tests.util import TEST_ENCRYPTION_KEY4
from gslib.tests.util import TEST_ENCRYPTION_KEY4_SHA256_B64
from gslib.tests.util import unittest
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
from gslib.utils.constants import UTF8
from gslib.utils.ls_helper import PrintFullInfoAboutObject
from gslib.utils.retry_util import Retry
from gslib.utils.system_util import IS_WINDOWS
from gslib.utils import shim_util
from six import add_move, MovedModule
from six.moves import mock
class TestLs(testcase.GsUtilIntegrationTestCase):
    """Integration tests for ls command."""

    def test_blank_ls(self):
        if not RUN_S3_TESTS:
            self.RunGsUtil(['ls'])

    def test_empty_bucket(self):
        bucket_uri = self.CreateBucket()
        self.AssertNObjectsInBucket(bucket_uri, 0)

    def test_empty_bucket_with_b(self):
        bucket_uri = self.CreateBucket()

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check1():
            stdout = self.RunGsUtil(['ls', '-b', suri(bucket_uri)], return_stdout=True)
            self.assertEqual('%s/\n' % suri(bucket_uri), stdout)
        _Check1()

    def test_bucket_with_Lb(self):
        """Tests ls -Lb."""
        bucket_uri = self.CreateBucket()
        stdout = self.RunGsUtil(['ls', '-Lb', suri(bucket_uri)], return_stdout=True)
        self.assertIn(suri(bucket_uri), stdout)
        self.assertNotIn('TOTAL:', stdout)
        self.RunGsUtil(['versioning', 'set', 'on', suri(bucket_uri)])
        self.RunGsUtil(['versioning', 'set', 'off', suri(bucket_uri)])
        stdout = self.RunGsUtil(['ls', '-Lb', suri(bucket_uri)], return_stdout=True)
        find_metageneration_re = re.compile('^\\s*Metageneration:\\s+(?P<metageneration_val>.+)$', re.MULTILINE)
        find_time_created_re = re.compile('^\\s*Time created:\\s+(?P<time_created_val>.+)$', re.MULTILINE)
        find_time_updated_re = re.compile('^\\s*Time updated:\\s+(?P<time_updated_val>.+)$', re.MULTILINE)
        metageneration_match = re.search(find_metageneration_re, stdout)
        time_created_match = re.search(find_time_created_re, stdout)
        time_updated_match = re.search(find_time_updated_re, stdout)
        if self.test_api == ApiSelector.XML:
            self.assertIsNone(metageneration_match)
            self.assertIsNone(time_created_match)
            self.assertIsNone(time_updated_match)
        elif self.test_api == ApiSelector.JSON:
            self.assertIsNotNone(metageneration_match)
            self.assertIsNotNone(time_created_match)
            self.assertIsNotNone(time_updated_match)
            time_created = time_created_match.group('time_created_val')
            time_created = time.strptime(time_created, '%a, %d %b %Y %H:%M:%S %Z')
            time_updated = time_updated_match.group('time_updated_val')
            time_updated = time.strptime(time_updated, '%a, %d %b %Y %H:%M:%S %Z')
            self.assertGreater(time_updated, time_created)
            self._AssertBucketPolicyOnly(False, stdout)

    def test_bucket_with_Lb_bucket_policy_only(self):
        if self.test_api == ApiSelector.JSON:
            bucket_uri = self.CreateBucket(bucket_policy_only=True, prefer_json_api=True)
            stdout = self.RunGsUtil(['ls', '-Lb', suri(bucket_uri)], return_stdout=True)
            self._AssertBucketPolicyOnly(True, stdout)

    def _AssertBucketPolicyOnly(self, value, stdout):
        bucket_policy_only_re = re.compile('^\\s*Bucket Policy Only enabled:\\s+(?P<bpo_val>.+)$', re.MULTILINE)
        bucket_policy_only_match = re.search(bucket_policy_only_re, stdout)
        bucket_policy_only_val = bucket_policy_only_match.group('bpo_val')
        self.assertEqual(str(value), bucket_policy_only_val)

    def test_bucket_with_lb(self):
        """Tests ls -lb."""
        bucket_uri = self.CreateBucket()

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check1():
            stdout = self.RunGsUtil(['ls', '-lb', suri(bucket_uri)], return_stdout=True)
            self.assertIn(suri(bucket_uri), stdout)
            self.assertNotIn('TOTAL:', stdout)
        _Check1()

    def test_bucket_list_wildcard(self):
        """Tests listing multiple buckets with a wildcard."""
        random_prefix = self.MakeRandomTestString()
        bucket1_name = self.MakeTempName('bucket', prefix=random_prefix)
        bucket2_name = self.MakeTempName('bucket', prefix=random_prefix)
        bucket1_uri = self.CreateBucket(bucket_name=bucket1_name)
        bucket2_uri = self.CreateBucket(bucket_name=bucket2_name)
        common_prefix = posixpath.commonprefix([suri(bucket1_uri), suri(bucket2_uri)])
        self.assertTrue(common_prefix.startswith('%s://%sgsutil-test-test-bucket-list-wildcard' % (self.default_provider, random_prefix)))
        wildcard = '%s*' % common_prefix

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check1():
            stdout = self.RunGsUtil(['ls', '-b', wildcard], return_stdout=True)
            expected = set([suri(bucket1_uri) + '/', suri(bucket2_uri) + '/'])
            actual = set(stdout.split())
            self.assertEqual(expected, actual)
        _Check1()

    def test_nonexistent_bucket_with_ls(self):
        """Tests a bucket that is known not to exist."""
        stderr = self.RunGsUtil(['ls', '-lb', 'gs://%s' % self.nonexistent_bucket_name], return_stderr=True, expected_status=1)
        self.assertIn('404', stderr)
        stderr = self.RunGsUtil(['ls', '-Lb', 'gs://%s' % self.nonexistent_bucket_name], return_stderr=True, expected_status=1)
        self.assertIn('404', stderr)
        stderr = self.RunGsUtil(['ls', '-b', 'gs://%s' % self.nonexistent_bucket_name], return_stderr=True, expected_status=1)
        self.assertIn('404', stderr)

    def test_list_missing_object(self):
        """Tests listing a non-existent object."""
        bucket_uri = self.CreateBucket()
        stderr = self.RunGsUtil(['ls', suri(bucket_uri, 'missing')], return_stderr=True, expected_status=1)
        self.assertIn('matched no objects', stderr)

    def test_with_one_object(self):
        bucket_uri = self.CreateBucket()
        obj_uri = self.CreateObject(bucket_uri=bucket_uri, contents=b'foo')

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check1():
            stdout = self.RunGsUtil(['ls', suri(bucket_uri)], return_stdout=True)
            self.assertEqual('%s\n' % obj_uri, stdout)
        _Check1()

    def location_redirect_test_helper(self, bucket_region, client_region):
        bucket_host = 's3.%s.amazonaws.com' % bucket_region
        client_host = 's3.%s.amazonaws.com' % client_region
        with SetBotoConfigForTest([('s3', 'host', bucket_host)]):
            bucket_uri = self.CreateBucket(location=bucket_region)
            obj_uri = self.CreateObject(bucket_uri=bucket_uri, contents=b'foo')

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check1(uri):
            stdout = self.RunGsUtil(['ls', uri], return_stdout=True)
            self.assertEqual('%s\n' % obj_uri, stdout)
        with SetBotoConfigForTest([('s3', 'host', client_host)]):
            _Check1(suri(bucket_uri))
            _Check1(suri(obj_uri))

    @SkipForGS('Only s3 V4 signatures error on location mismatches.')
    def test_400_location_redirect(self):
        self.location_redirect_test_helper('ap-east-1', 'us-east-2')

    @SkipForGS('Only s3 V4 signatures error on location mismatches.')
    def test_301_location_redirect(self):
        self.location_redirect_test_helper('eu-west-1', 'us-east-2')

    @SkipForS3('Not relevant for S3')
    @SkipForJSON('Only the XML API supports changing the calling format.')
    def test_default_gcs_calling_format_is_path_style(self):
        bucket_uri = self.CreateBucket()
        object_uri = self.CreateObject(bucket_uri=bucket_uri, contents=b'foo')
        stderr = self.RunGsUtil(['-D', 'ls', suri(object_uri)], return_stdout=True)
        self.assertIn('Host: storage.googleapis.com', stderr)

    @SkipForS3('Not relevant for S3')
    @SkipForJSON('Only the XML API supports changing the calling format.')
    def test_gcs_calling_format_is_configurable(self):
        bucket_uri = self.CreateBucket()
        object_uri = self.CreateObject(bucket_uri=bucket_uri, contents=b'foo')
        custom_calling_format = 'boto.s3.connection.SubdomainCallingFormat'
        with SetBotoConfigForTest([('s3', 'calling_format', custom_calling_format)]):
            stderr = self.RunGsUtil(['-D', 'ls', suri(object_uri)], return_stdout=True)
        self.assertIn('Host: %s.storage.googleapis.com' % bucket_uri.bucket_name, stderr)

    @SkipForXML('Credstore file gets created only for json API')
    def test_credfile_lock_permissions(self):
        tmpdir = self.CreateTempDir()
        filepath = os.path.join(tmpdir, 'credstore2')
        option = 'GSUtil:state_dir={}'.format(tmpdir)
        bucket_uri = self.CreateBucket()
        obj_uri = self.CreateObject(bucket_uri=bucket_uri, contents=b'foo')

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check1():
            stdout = self.RunGsUtil(['-o', option, 'ls', suri(bucket_uri)], return_stdout=True)
            self.assertEqual('%s\n' % obj_uri, stdout)
            if os.name == 'posix':
                self.assertTrue(os.path.exists(filepath))
                mode = oct(stat.S_IMODE(os.stat(filepath).st_mode))
                self.assertEqual(oct(384), mode)
        _Check1()

    def test_one_object_with_l(self):
        """Tests listing one object with -l."""
        obj_uri = self.CreateObject(contents=b'foo')
        stdout = self.RunGsUtil(['ls', '-l', suri(obj_uri)], return_stdout=True)
        output_items = stdout.split()
        self.assertTrue(output_items[0].isdigit())
        time.strptime(stdout.split()[1], '%Y-%m-%dT%H:%M:%SZ')
        self.assertEqual(output_items[2], suri(obj_uri))

    def test_one_object_with_L(self):
        """Tests listing one object with -L."""
        obj_uri = self.CreateObject(contents=b'foo')
        time.sleep(2)
        self.RunGsUtil(['setmeta', '-h', 'x-goog-meta-foo:bar', suri(obj_uri)])
        find_time_created_re = re.compile('^\\s*Creation time:\\s+(?P<time_created_val>.+)$', re.MULTILINE)
        find_time_updated_re = re.compile('^\\s*Update time:\\s+(?P<time_updated_val>.+)$', re.MULTILINE)
        stdout = self.RunGsUtil(['ls', '-L', suri(obj_uri)], return_stdout=True)
        time_created_match = re.search(find_time_created_re, stdout)
        time_updated_match = re.search(find_time_updated_re, stdout)
        time_created = time_created_match.group('time_created_val')
        self.assertIsNotNone(time_created)
        time_created = time.strptime(time_created, '%a, %d %b %Y %H:%M:%S %Z')
        if self.test_api == ApiSelector.XML:
            self.assertIsNone(time_updated_match)
        elif self.test_api == ApiSelector.JSON:
            time_updated = time_updated_match.group('time_updated_val')
            self.assertIsNotNone(time_updated)
            time_updated = time.strptime(time_updated, '%a, %d %b %Y %H:%M:%S %Z')
            self.assertGreater(time_updated, time_created)

    @SkipForS3('Integration test utils only support GCS JSON for versioning.')
    @SkipForXML('Integration test utils only support GCS JSON for versioning.')
    def test_one_object_with_generation(self):
        """Tests listing one object by generation when multiple versions exist."""
        bucket = self.CreateBucketJson(versioning_enabled=True)
        object1 = self.CreateObjectJson(bucket_name=bucket.name, contents=b'1')
        object2 = self.CreateObjectJson(bucket_name=bucket.name, object_name=object1.name, contents=b'2')
        object_url_string1 = 'gs://{}/{}#{}'.format(object1.bucket, object1.name, object1.generation)
        object_url_string2 = 'gs://{}/{}#{}'.format(object2.bucket, object2.name, object2.generation)
        stdout = self.RunGsUtil(['ls', object_url_string2], return_stdout=True)
        self.assertNotIn(object_url_string1, stdout)
        self.assertIn(object_url_string2, stdout)

    def test_subdir(self):
        """Tests listing a bucket subdirectory."""
        bucket_uri = self.CreateBucket(test_objects=1)
        k1_uri = self.StorageUriCloneReplaceName(bucket_uri, 'foo')
        self.StorageUriSetContentsFromString(k1_uri, 'baz')
        k2_uri = self.StorageUriCloneReplaceName(bucket_uri, 'dir/foo')
        self.StorageUriSetContentsFromString(k2_uri, 'bar')

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check1():
            stdout = self.RunGsUtil(['ls', '%s/dir' % suri(bucket_uri)], return_stdout=True)
            self.assertEqual('%s\n' % suri(k2_uri), stdout)
            stdout = self.RunGsUtil(['ls', suri(k1_uri)], return_stdout=True)
            self.assertEqual('%s\n' % suri(k1_uri), stdout)
        _Check1()

    def test_subdir_nocontents(self):
        """Tests listing a bucket subdirectory using -d.

    Result will display subdirectory names instead of contents. Uses a wildcard
    to show multiple matching subdirectories.
    """
        bucket_uri = self.CreateBucket(test_objects=1)
        k1_uri = self.StorageUriCloneReplaceName(bucket_uri, 'foo')
        self.StorageUriSetContentsFromString(k1_uri, 'baz')
        k2_uri = self.StorageUriCloneReplaceName(bucket_uri, 'dir/foo')
        self.StorageUriSetContentsFromString(k2_uri, 'bar')
        k3_uri = self.StorageUriCloneReplaceName(bucket_uri, 'dir/foo2')
        self.StorageUriSetContentsFromString(k3_uri, 'foo')
        k4_uri = self.StorageUriCloneReplaceName(bucket_uri, 'dir2/foo3')
        self.StorageUriSetContentsFromString(k4_uri, 'foo2')

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check1():
            stdout = self.RunGsUtil(['ls', '-d', '%s/dir*' % suri(bucket_uri)], return_stdout=True)
            self.assertEqual('%s/dir/\n%s/dir2/\n' % (suri(bucket_uri), suri(bucket_uri)), stdout)
            stdout = self.RunGsUtil(['ls', suri(k1_uri)], return_stdout=True)
            self.assertEqual('%s\n' % suri(k1_uri), stdout)
        _Check1()

    def test_versioning(self):
        """Tests listing a versioned bucket."""
        bucket1_uri = self.CreateBucket(test_objects=1)
        bucket2_uri = self.CreateVersionedBucket(test_objects=1)
        self.AssertNObjectsInBucket(bucket1_uri, 1, versioned=True)
        bucket_list = list(bucket1_uri.list_bucket())
        objuri = [self.StorageUriCloneReplaceKey(bucket1_uri, key).versionless_uri for key in bucket_list][0]
        self.RunGsUtil(['cp', objuri, suri(bucket2_uri)])
        self.RunGsUtil(['cp', objuri, suri(bucket2_uri)])

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check2():
            stdout = self.RunGsUtil(['ls', '-a', suri(bucket2_uri)], return_stdout=True)
            self.assertNumLines(stdout, 3)
            stdout = self.RunGsUtil(['ls', '-la', suri(bucket2_uri)], return_stdout=True)
            self.assertIn('%s#' % self.StorageUriCloneReplaceName(bucket2_uri, bucket_list[0].name), stdout)
            self.assertIn('metageneration=', stdout)
        _Check2()

    def test_etag(self):
        """Tests that listing an object with an etag."""
        bucket_uri = self.CreateBucket()
        obj_uri = self.CreateObject(bucket_uri=bucket_uri, contents=b'foo')
        etag = obj_uri.get_key().etag.strip('"\'')

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check1():
            stdout = self.RunGsUtil(['ls', '-l', suri(bucket_uri)], return_stdout=True)
            if self.test_api == ApiSelector.XML:
                self.assertNotIn(etag, stdout)
            else:
                self.assertNotIn('etag=', stdout)
        _Check1()

        def _Check2():
            stdout = self.RunGsUtil(['ls', '-le', suri(bucket_uri)], return_stdout=True)
            if self.test_api == ApiSelector.XML:
                self.assertIn(etag, stdout)
            else:
                self.assertIn('etag=', stdout)
        _Check2()

        def _Check3():
            stdout = self.RunGsUtil(['ls', '-ale', suri(bucket_uri)], return_stdout=True)
            if self.test_api == ApiSelector.XML:
                self.assertIn(etag, stdout)
            else:
                self.assertIn('etag=', stdout)
        _Check3()

    def test_labels(self):
        """Tests listing on a bucket with a label/tagging configuration."""
        bucket_uri = self.CreateBucket()
        bucket_suri = suri(bucket_uri)
        stdout = self.RunGsUtil(['ls', '-Lb', bucket_suri], return_stdout=True)
        self.assertRegex(stdout, 'Labels:\\s+None')
        self.RunGsUtil(['label', 'ch', '-l', 'labelkey:labelvalue', bucket_suri], force_gsutil=True)
        stdout = self.RunGsUtil(['ls', '-Lb', bucket_suri], return_stdout=True)
        label_regex = re.compile('Labels:\\s+\\{\\s+"labelkey":\\s+"labelvalue"\\s+\\}', re.MULTILINE)
        self.assertRegex(stdout, label_regex)

    @SkipForS3('S3 bucket configuration values are not supported via ls.')
    def test_location_constraint(self):
        """Tests listing a bucket with location constraint."""
        bucket_uri = self.CreateBucket()
        bucket_suri = suri(bucket_uri)
        stdout = self.RunGsUtil(['ls', '-lb', bucket_suri], return_stdout=True)
        self.assertNotIn('Location constraint:', stdout)
        stdout = self.RunGsUtil(['ls', '-Lb', bucket_suri], return_stdout=True)
        self.assertRegex(stdout, 'Location constraint:\\s+\\S')

    @unittest.skip('b/135700569')
    @SkipForXML('Location type not available when using the GCS XML API.')
    @SkipForS3('Location type not printed for S3 buckets.')
    def test_location_type(self):
        """Tests listing a bucket with location constraint."""
        bucket_uri = self.CreateBucket()
        bucket_suri = suri(bucket_uri)
        stdout = self.RunGsUtil(['ls', '-lb', bucket_suri], return_stdout=True)
        self.assertNotIn('Location type:', stdout)
        stdout = self.RunGsUtil(['ls', '-Lb', bucket_suri], return_stdout=True)
        self.assertRegex(stdout, 'Location type:\\s+\\S')

    @SkipForS3('S3 bucket configuration values are not supported via ls.')
    def test_logging(self):
        """Tests listing a bucket with logging config."""
        bucket_uri = self.CreateBucket()
        bucket_suri = suri(bucket_uri)
        stdout = self.RunGsUtil(['ls', '-lb', bucket_suri], return_stdout=True)
        self.assertNotIn('Logging configuration', stdout)
        spacing = '       ' if self._use_gcloud_storage else '\t\t'
        stdout = self.RunGsUtil(['ls', '-Lb', bucket_suri], return_stdout=True)
        self.assertIn('Logging configuration:{}None'.format(spacing), stdout)
        self.RunGsUtil(['logging', 'set', 'on', '-b', bucket_suri, bucket_suri])
        stdout = self.RunGsUtil(['ls', '-Lb', bucket_suri], return_stdout=True)
        self.assertIn('Logging configuration:{}Present'.format(spacing), stdout)
        self.RunGsUtil(['logging', 'set', 'off', bucket_suri])
        stdout = self.RunGsUtil(['ls', '-Lb', bucket_suri], return_stdout=True)
        self.assertIn('Logging configuration:{}None'.format(spacing), stdout)

    @SkipForS3('S3 bucket configuration values are not supported via ls.')
    def test_web(self):
        """Tests listing a bucket with website config."""
        bucket_uri = self.CreateBucket()
        bucket_suri = suri(bucket_uri)
        stdout = self.RunGsUtil(['ls', '-lb', bucket_suri], return_stdout=True)
        self.assertNotIn('Website configuration', stdout)
        spacing = '       ' if self._use_gcloud_storage else '\t\t'
        stdout = self.RunGsUtil(['ls', '-Lb', bucket_suri], return_stdout=True)
        self.assertIn('Website configuration:{}None'.format(spacing), stdout)
        self.RunGsUtil(['web', 'set', '-m', 'google.com', bucket_suri])
        stdout = self.RunGsUtil(['ls', '-Lb', bucket_suri], return_stdout=True)
        self.assertIn('Website configuration:{}Present'.format(spacing), stdout)
        self.RunGsUtil(['web', 'set', bucket_suri])
        stdout = self.RunGsUtil(['ls', '-Lb', bucket_suri], return_stdout=True)
        self.assertIn('Website configuration:{}None'.format(spacing), stdout)

    @SkipForS3('S3 bucket configuration values are not supported via ls.')
    @SkipForXML('Requester Pays is not supported for the XML API.')
    def test_requesterpays(self):
        """Tests listing a bucket with requester pays (billing) config."""
        bucket_uri = self.CreateBucket()
        bucket_suri = suri(bucket_uri)
        spacing = '      ' if self._use_gcloud_storage else '\t\t'
        stdout = self.RunGsUtil(['ls', '-lb', bucket_suri], return_stdout=True)
        self.assertNotIn('Requester Pays enabled', stdout)
        stdout = self.RunGsUtil(['ls', '-Lb', bucket_suri], return_stdout=True)
        self.assertIn('Requester Pays enabled:{}None'.format(spacing), stdout)
        self.RunGsUtil(['requesterpays', 'set', 'on', bucket_suri])
        stdout = self.RunGsUtil(['ls', '-Lb', bucket_suri], return_stdout=True)
        self.assertIn('Requester Pays enabled:{}True'.format(spacing), stdout)
        self.RunGsUtil(['requesterpays', 'set', 'off', bucket_suri])
        stdout = self.RunGsUtil(['ls', '-Lb', bucket_suri], return_stdout=True)
        self.assertIn('Requester Pays enabled:{}False'.format(spacing), stdout)

    def test_list_sizes(self):
        """Tests various size listing options."""
        bucket_uri = self.CreateBucket()
        self.CreateObject(bucket_uri=bucket_uri, contents=b'x' * 2048)

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check1():
            stdout = self.RunGsUtil(['ls', '-l', suri(bucket_uri)], return_stdout=True)
            self.assertIn('2048', stdout)
        _Check1()

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check2():
            stdout = self.RunGsUtil(['ls', '-L', suri(bucket_uri)], return_stdout=True)
            self.assertIn('2048', stdout)
        _Check2()

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check3():
            stdout = self.RunGsUtil(['ls', '-al', suri(bucket_uri)], return_stdout=True)
            self.assertIn('2048', stdout)
        _Check3()

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check4():
            stdout = self.RunGsUtil(['ls', '-lh', suri(bucket_uri)], return_stdout=True)
            self.assertIn('2 KiB', stdout)
        _Check4()

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check5():
            stdout = self.RunGsUtil(['ls', '-alh', suri(bucket_uri)], return_stdout=True)
            self.assertIn('2 KiB', stdout)
        _Check5()

    @unittest.skipIf(IS_WINDOWS, 'Unicode handling on Windows requires mods to site-packages')
    def test_list_unicode_filename(self):
        """Tests listing an object with a unicode filename."""
        object_name = u'Аудиоархив'
        bucket_uri = self.CreateVersionedBucket()
        key_uri = self.CreateObject(bucket_uri=bucket_uri, contents=b'foo', object_name=object_name)
        self.AssertNObjectsInBucket(bucket_uri, 1, versioned=True)
        stdout = self.RunGsUtil(['ls', '-ael', suri(key_uri)], return_stdout=True)
        self.assertIn(object_name, stdout)
        if self.default_provider == 'gs':
            self.assertIn(str(key_uri.generation), stdout)
            self.assertIn('metageneration=%s' % key_uri.get_key().metageneration, stdout)
            if self.test_api == ApiSelector.XML:
                self.assertIn(key_uri.get_key().etag.strip('"\''), stdout)
            else:
                self.assertIn('etag=', stdout)
        elif self.default_provider == 's3':
            self.assertIn(key_uri.version_id, stdout)
            self.assertIn(key_uri.get_key().etag.strip('"\''), stdout)

    def test_list_acl(self):
        """Tests that long listing includes an ACL."""
        key_uri = self.CreateObject(contents=b'foo')
        stdout = self.RunGsUtil(['ls', '-L', suri(key_uri)], return_stdout=True)
        self.assertIn('ACL:', stdout)
        self.assertNotIn('ACCESS DENIED', stdout)

    def test_list_gzip_content_length(self):
        """Tests listing a gzipped object."""
        file_size = 10000
        file_contents = b'x' * file_size
        fpath = self.CreateTempFile(contents=file_contents, file_name='foo.txt')
        key_uri = self.CreateObject()
        self.RunGsUtil(['cp', '-z', 'txt', suri(fpath), suri(key_uri)])

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check1():
            stdout = self.RunGsUtil(['ls', '-L', suri(key_uri)], return_stdout=True)
            self.assertRegex(stdout, 'Content-Encoding:\\s+gzip')
            find_content_length_re = 'Content-Length:\\s+(?P<num>\\d)'
            self.assertRegex(stdout, find_content_length_re)
            m = re.search(find_content_length_re, stdout)
            content_length = int(m.group('num'))
            self.assertGreater(content_length, 0)
            self.assertLess(content_length, file_size)
        _Check1()

    def test_output_chopped(self):
        """Tests that gsutil still succeeds with a truncated stdout."""
        bucket_uri = self.CreateBucket(test_objects=2)
        gsutil_cmd = [sys.executable, '-u', gslib.GSUTIL_PATH, 'ls', suri(bucket_uri)]
        p = subprocess.Popen(gsutil_cmd, stdout=subprocess.PIPE, bufsize=0)
        p.stdout.close()
        p.wait()
        self.assertEqual(p.returncode, 0)

    @SkipForS3('Boto lib required for S3 does not handle paths starting with slash.')
    def test_recursive_list_slash_only(self):
        """Tests listing an object with a trailing slash."""
        bucket_uri = self.CreateBucket()
        self.CreateObject(bucket_uri=bucket_uri, object_name='/', contents=b'foo')
        self.AssertNObjectsInBucket(bucket_uri, 1)
        stdout = self.RunGsUtil(['ls', '-R', suri(bucket_uri)], return_stdout=True)
        self.assertIn(suri(bucket_uri) + '/', stdout)

    def test_recursive_list_trailing_slash(self):
        """Tests listing an object with a trailing slash."""
        bucket_uri = self.CreateBucket()
        self.CreateObject(bucket_uri=bucket_uri, object_name='foo/', contents=b'foo')
        self.AssertNObjectsInBucket(bucket_uri, 1)
        stdout = self.RunGsUtil(['ls', '-R', suri(bucket_uri)], return_stdout=True)
        self.assertIn(suri(bucket_uri) + '/foo/', stdout)

    @SkipForS3('Boto lib required for S3 does not handle paths starting with slash.')
    def test_recursive_list_trailing_two_slash(self):
        """Tests listing an object with two trailing slashes."""
        bucket_uri = self.CreateBucket()
        self.CreateObject(bucket_uri=bucket_uri, object_name='//', contents=b'foo')
        self.AssertNObjectsInBucket(bucket_uri, 1)
        stdout = self.RunGsUtil(['ls', '-R', suri(bucket_uri)], return_stdout=True)
        self.assertIn(suri(bucket_uri) + '//', stdout)

    def test_wildcard_prefix(self):
        """Tests that an object name with a wildcard does not infinite loop."""
        bucket_uri = self.CreateBucket()
        wildcard_folder_object = 'wildcard*/'
        object_matching_folder = 'wildcard10/foo'
        self.CreateObject(bucket_uri=bucket_uri, object_name=wildcard_folder_object, contents=b'foo')
        self.CreateObject(bucket_uri=bucket_uri, object_name=object_matching_folder, contents=b'foo')
        self.AssertNObjectsInBucket(bucket_uri, 2)
        stderr = self.RunGsUtil(['ls', suri(bucket_uri, 'wildcard*')], return_stderr=True, expected_status=1)
        if self._use_gcloud_storage:
            warning_message = 'Cloud folders named with wildcards are not supported. API returned {}/wildcard*/'
        else:
            warning_message = 'Cloud folder {}/wildcard*/ contains a wildcard'
        self.assertIn(warning_message.format(suri(bucket_uri)), stderr)

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check():
            stdout = self.RunGsUtil(['ls', '-l', suri(bucket_uri, '**')], return_stdout=True)
            self.assertNumLines(stdout, 3)
        _Check()

    @SkipForS3('S3 anonymous access is not supported.')
    def test_get_object_without_list_bucket_permission(self):
        bucket_uri = self.CreateBucket()
        object_uri = self.CreateObject(bucket_uri=bucket_uri, object_name='permitted', contents=b'foo')
        self.RunGsUtil(['acl', 'set', 'public-read', suri(object_uri)], force_gsutil=True)
        with self.SetAnonymousBotoCreds():
            stdout = self.RunGsUtil(['ls', '-L', suri(object_uri)], return_stdout=True)
            self.assertIn(suri(object_uri), stdout)

    @SkipForS3('S3 customer-supplied encryption keys are not supported.')
    def test_list_encrypted_object(self):
        if self.test_api == ApiSelector.XML:
            return unittest.skip('gsutil does not support encryption with the XML API')
        object_uri = self.CreateObject(object_name='foo', contents=TEST_ENCRYPTION_CONTENT1, encryption_key=TEST_ENCRYPTION_KEY1)
        with SetBotoConfigForTest([('GSUtil', 'encryption_key', TEST_ENCRYPTION_KEY1)]):

            @Retry(AssertionError, tries=3, timeout_secs=1)
            def _ListExpectDecrypted():
                stdout = self.RunGsUtil(['ls', '-L', suri(object_uri)], return_stdout=True)
                self.assertIn(TEST_ENCRYPTION_CONTENT1_MD5, stdout)
                self.assertIn(TEST_ENCRYPTION_CONTENT1_CRC32C, stdout)
                self.assertIn(TEST_ENCRYPTION_KEY1_SHA256_B64.decode('ascii'), stdout)
            _ListExpectDecrypted()

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _ListExpectEncrypted():
            stdout = self.RunGsUtil(['ls', '-L', suri(object_uri)], return_stdout=True)
            self.assertNotIn(TEST_ENCRYPTION_CONTENT1_MD5, stdout)
            self.assertNotIn(TEST_ENCRYPTION_CONTENT1_CRC32C, stdout)
            self.assertIn('encrypted', stdout)
            self.assertIn(TEST_ENCRYPTION_KEY1_SHA256_B64.decode('ascii'), stdout)
        _ListExpectEncrypted()
        with SetBotoConfigForTest([('GSUtil', 'encryption_key', TEST_ENCRYPTION_KEY2)]):
            _ListExpectEncrypted()

    @SkipForS3('S3 customer-supplied encryption keys are not supported.')
    def test_list_mixed_encryption(self):
        """Tests listing objects with various encryption interactions."""
        bucket_uri = self.CreateBucket()
        self.CreateObject(bucket_uri=bucket_uri, object_name='foo', contents=TEST_ENCRYPTION_CONTENT1, encryption_key=TEST_ENCRYPTION_KEY1)
        self.CreateObject(bucket_uri=bucket_uri, object_name='foo2', contents=TEST_ENCRYPTION_CONTENT2, encryption_key=TEST_ENCRYPTION_KEY2)
        self.CreateObject(bucket_uri=bucket_uri, object_name='foo3', contents=TEST_ENCRYPTION_CONTENT3, encryption_key=TEST_ENCRYPTION_KEY3)
        self.CreateObject(bucket_uri=bucket_uri, object_name='foo4', contents=TEST_ENCRYPTION_CONTENT4, encryption_key=TEST_ENCRYPTION_KEY4)
        self.CreateObject(bucket_uri=bucket_uri, object_name='foo5', contents=TEST_ENCRYPTION_CONTENT5)
        with SetBotoConfigForTest([('GSUtil', 'encryption_key', TEST_ENCRYPTION_KEY1), ('GSUtil', 'decryption_key1', TEST_ENCRYPTION_KEY3), ('GSUtil', 'decryption_key2', TEST_ENCRYPTION_KEY4)]):

            @Retry(AssertionError, tries=3, timeout_secs=1)
            def _ListExpectMixed():
                """Validates object listing."""
                stdout = self.RunGsUtil(['ls', '-L', suri(bucket_uri)], return_stdout=True)
                self.assertIn(TEST_ENCRYPTION_CONTENT1_MD5, stdout)
                self.assertIn(TEST_ENCRYPTION_CONTENT1_CRC32C, stdout)
                self.assertIn(TEST_ENCRYPTION_KEY1_SHA256_B64.decode('ascii'), stdout)
                self.assertNotIn(TEST_ENCRYPTION_CONTENT2_MD5, stdout)
                self.assertNotIn(TEST_ENCRYPTION_CONTENT2_CRC32C, stdout)
                self.assertIn('encrypted', stdout)
                self.assertIn(TEST_ENCRYPTION_KEY2_SHA256_B64.decode('ascii'), stdout)
                self.assertIn(TEST_ENCRYPTION_CONTENT3_MD5, stdout)
                self.assertIn(TEST_ENCRYPTION_CONTENT3_CRC32C, stdout)
                self.assertIn(TEST_ENCRYPTION_KEY3_SHA256_B64.decode('ascii'), stdout)
                self.assertIn(TEST_ENCRYPTION_CONTENT4_MD5, stdout)
                self.assertIn(TEST_ENCRYPTION_CONTENT4_CRC32C, stdout)
                self.assertIn(TEST_ENCRYPTION_KEY4_SHA256_B64.decode('ascii'), stdout)
                self.assertIn(TEST_ENCRYPTION_CONTENT5_MD5, stdout)
                self.assertIn(TEST_ENCRYPTION_CONTENT5_CRC32C, stdout)
            _ListExpectMixed()

    def test_non_ascii_project_fails(self):
        stderr = self.RunGsUtil(['ls', '-p', 'ã', 'gs://fobarbaz'], expected_status=1, return_stderr=True)
        self.assertIn('Invalid non-ASCII', stderr)

    def set_default_kms_key_on_bucket(self, bucket_uri):
        keyring_fqn = self.kms_api.CreateKeyRing(PopulateProjectId(None), testcase.KmsTestingResources.KEYRING_NAME, location=testcase.KmsTestingResources.KEYRING_LOCATION)
        key_fqn = self.kms_api.CreateCryptoKey(keyring_fqn, testcase.KmsTestingResources.CONSTANT_KEY_NAME)
        self.RunGsUtil(['kms', 'encryption', '-k', key_fqn, suri(bucket_uri)])
        return key_fqn

    @SkipForXML(KMS_XML_SKIP_MSG)
    @SkipForS3(KMS_XML_SKIP_MSG)
    def test_default_kms_key_listed_for_bucket(self):
        bucket_uri = self.CreateBucket()
        stdout = self.RunGsUtil(['ls', '-Lb', suri(bucket_uri)], return_stdout=True)
        self.assertRegex(stdout, 'Default KMS key:\\s+None')
        key_fqn = self.set_default_kms_key_on_bucket(bucket_uri)
        stdout = self.RunGsUtil(['ls', '-Lb', suri(bucket_uri)], return_stdout=True)
        self.assertRegex(stdout, 'Default KMS key:\\s+%s' % key_fqn)

    @SkipForXML(KMS_XML_SKIP_MSG)
    @SkipForS3(KMS_XML_SKIP_MSG)
    def test_kms_key_listed_for_kms_encrypted_object(self):
        bucket_uri = self.CreateBucket()
        key_fqn = self.set_default_kms_key_on_bucket(bucket_uri)
        obj_uri = self.CreateObject(bucket_uri=bucket_uri, object_name='foo', contents=b'foo', kms_key_name=key_fqn)
        stdout = self.RunGsUtil(['ls', '-L', suri(obj_uri)], return_stdout=True)
        self.assertRegex(stdout, 'KMS key:\\s+%s' % key_fqn)

    @SkipForXML(BUCKET_LOCK_SKIP_MSG)
    @SkipForS3(BUCKET_LOCK_SKIP_MSG)
    def test_list_retention_policy(self):
        bucket_uri = self.CreateBucketWithRetentionPolicy(retention_period_in_seconds=1)
        stdout = self.RunGsUtil(['ls', '-Lb', suri(bucket_uri)], return_stdout=True)
        self.assertRegex(stdout, 'Retention Policy\\:\\s*Present')
        self.RunGsUtil(['retention', 'clear', suri(bucket_uri)])
        stdout = self.RunGsUtil(['ls', '-Lb', suri(bucket_uri)], return_stdout=True)
        self.assertNotRegex(stdout, 'Retention Policy:')

    @SkipForXML(BUCKET_LOCK_SKIP_MSG)
    @SkipForS3(BUCKET_LOCK_SKIP_MSG)
    def test_list_default_event_based_hold(self):
        bucket_uri = self.CreateBucket()
        self.RunGsUtil(['retention', 'event-default', 'set', suri(bucket_uri)])
        stdout = self.RunGsUtil(['ls', '-Lb', suri(bucket_uri)], return_stdout=True)
        self.assertRegex(stdout, 'Default Event-Based Hold:\\t* *True')
        self.RunGsUtil(['retention', 'event-default', 'release', suri(bucket_uri)])
        stdout = self.RunGsUtil(['ls', '-Lb', suri(bucket_uri)], return_stdout=True)
        self.assertNotRegex(stdout, 'Default Event-Based Hold')

    @SkipForXML(BUCKET_LOCK_SKIP_MSG)
    @SkipForS3(BUCKET_LOCK_SKIP_MSG)
    def test_list_temporary_hold(self):
        object_uri = self.CreateObject(contents=b'content')
        self.RunGsUtil(['retention', 'temp', 'set', suri(object_uri)])
        stdout = self.RunGsUtil(['ls', '-L', suri(object_uri)], return_stdout=True)
        self.assertRegex(stdout, 'Temporary Hold')
        self.RunGsUtil(['retention', 'temp', 'release', suri(object_uri)])
        stdout = self.RunGsUtil(['ls', '-L', suri(object_uri)], return_stdout=True)
        self.assertNotRegex(stdout, 'Temporary Hold')

    @SkipForXML(BUCKET_LOCK_SKIP_MSG)
    @SkipForS3(BUCKET_LOCK_SKIP_MSG)
    def test_list_event_based_hold(self):
        object_uri = self.CreateObject(contents=b'content')
        self.RunGsUtil(['retention', 'event', 'set', suri(object_uri)])
        stdout = self.RunGsUtil(['ls', '-L', suri(object_uri)], return_stdout=True)
        self.assertRegex(stdout, 'Event-Based Hold')
        self.RunGsUtil(['retention', 'event', 'release', suri(object_uri)])
        stdout = self.RunGsUtil(['ls', '-L', suri(object_uri)], return_stdout=True)
        self.assertNotRegex(stdout, 'Event-Based Hold')

    @SkipForXML('public access prevention is not supported for the XML API.')
    @SkipForS3('public access prevention is not supported for S3 buckets.')
    def test_list_public_access_prevention(self):
        bucket_uri = self.CreateBucket()
        stdout = self.RunGsUtil(['ls', '-Lb', suri(bucket_uri)], return_stdout=True)
        self.assertRegex(stdout, 'Public access prevention:\\s*(unspecified|inherited)')
        self.RunGsUtil(['pap', 'set', 'enforced', suri(bucket_uri)])
        stdout = self.RunGsUtil(['ls', '-Lb', suri(bucket_uri)], return_stdout=True)
        self.assertRegex(stdout, 'Public access prevention:\\s*enforced')

    @SkipForXML('RPO is not supported for the XML API.')
    @SkipForS3('RPO is not supported for S3 buckets.')
    def test_list_Lb_displays_rpo(self):
        bucket_uri = self.CreateBucket(location='nam4')
        stdout = self.RunGsUtil(['ls', '-Lb', suri(bucket_uri)], return_stdout=True)
        self.RunGsUtil(['rpo', 'set', 'ASYNC_TURBO', suri(bucket_uri)])
        stdout = self.RunGsUtil(['ls', '-Lb', suri(bucket_uri)], return_stdout=True)
        self.assertRegex(stdout, 'RPO:\\t\\t\\t\\tASYNC_TURBO')

    @SkipForXML('Custom Dual Region is not supported for the XML API.')
    @SkipForS3('Custom Dual Region is not supported for S3 buckets.')
    def test_list_Lb_displays_custom_dual_region_placement_info(self):
        bucket_name = 'gs://' + self.MakeTempName('bucket')
        self.RunGsUtil(['mb', '--placement', 'us-central1,us-west1', bucket_name], expected_status=0)
        stdout = self.RunGsUtil(['ls', '-Lb', bucket_name], return_stdout=True)
        self.assertRegex(stdout, "Placement locations:\\t\\t\\['US-CENTRAL1', 'US-WEST1'\\]")

    @SkipForXML('Autoclass is not supported for the XML API.')
    @SkipForS3('Autoclass is not supported for S3 buckets.')
    def test_list_autoclass(self):
        bucket_uri = self.CreateBucket()
        stdout = self.RunGsUtil(['ls', '-Lb', suri(bucket_uri)], return_stdout=True)
        self.assertNotIn('Autoclass', stdout)
        self.RunGsUtil(['autoclass', 'set', 'on', suri(bucket_uri)])
        stdout = self.RunGsUtil(['ls', '-Lb', suri(bucket_uri)], return_stdout=True)
        self.assertRegex(stdout, 'Autoclass:\\t*Enabled on .+')