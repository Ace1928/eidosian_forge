from __future__ import absolute_import
import re
import gslib.tests.testcase as testcase
from gslib import exception
from gslib.tests.testcase.integration_testcase import SkipForGS
from gslib.tests.testcase.integration_testcase import SkipForJSON
from gslib.tests.testcase.integration_testcase import SkipForXML
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import SetBotoConfigForTest
class TestAutoclassE2E(testcase.GsUtilIntegrationTestCase):
    """E2E tests for autoclass command."""
    _set_autoclass_cmd = ['autoclass', 'set']
    _get_autoclass_cmd = ['autoclass', 'get']

    @SkipForXML('Autoclass only runs on GCS JSON API.')
    def test_off_on_default_buckets(self):
        bucket_uri = self.CreateBucket()
        stdout = self.RunGsUtil(self._get_autoclass_cmd + [suri(bucket_uri)], return_stdout=True)
        self.assertRegex(stdout, 'Enabled: False')
        self.assertRegex(stdout, 'Toggle Time: None')

    @SkipForXML('Autoclass only runs on GCS JSON API.')
    def test_turning_on_and_off(self):
        bucket_uri = self.CreateBucket()
        stdout, stderr = self.RunGsUtil(self._set_autoclass_cmd + ['on', suri(bucket_uri)], return_stdout=True, return_stderr=True)
        if self._use_gcloud_storage:
            self.assertRegex(stderr, re.escape('Updating {}'.format(str(bucket_uri))))
        else:
            self.assertRegex(stdout, re.escape('Setting Autoclass on for {}\n'.format(str(bucket_uri).rstrip('/'))))
        stdout = self.RunGsUtil(self._get_autoclass_cmd + [suri(bucket_uri)], return_stdout=True)
        self.assertRegex(stdout, 'Enabled: True')
        self.assertRegex(stdout, 'Toggle Time: \\d+')
        stdout, stderr = self.RunGsUtil(self._set_autoclass_cmd + ['off', suri(bucket_uri)], return_stdout=True, return_stderr=True)
        if self._use_gcloud_storage:
            self.assertRegex(stderr, re.escape('Updating {}'.format(str(bucket_uri))))
        else:
            self.assertRegex(stdout, re.escape('Setting Autoclass off for {}\n'.format(str(bucket_uri).rstrip('/'))))
        stdout = self.RunGsUtil(self._get_autoclass_cmd + [suri(bucket_uri)], return_stdout=True)
        self.assertRegex(stdout, 'Enabled: False')
        self.assertRegex(stdout, 'Toggle Time: \\d+')

    @SkipForXML('Autoclass only runs on GCS JSON API.')
    def test_multiple_buckets(self):
        bucket_uri1 = self.CreateBucket()
        bucket_uri2 = self.CreateBucket()
        stdout = self.RunGsUtil(self._get_autoclass_cmd + [suri(bucket_uri1), suri(bucket_uri2)], return_stdout=True)
        output_regex = '{}:\\n  Enabled: False\\n  Toggle Time: None\\n{}:\\n  Enabled: False\\n  Toggle Time: None'.format(suri(bucket_uri1), suri(bucket_uri2))
        self.assertRegex(stdout, output_regex)

    @SkipForJSON('Testing XML only behavior.')
    def test_xml_fails(self):
        boto_config_hmac_auth_only = [('Credentials', 'gs_oauth2_refresh_token', None), ('Credentials', 'gs_service_client_id', None), ('Credentials', 'gs_service_key_file', None), ('Credentials', 'gs_service_key_file_password', None), ('Credentials', 'gs_access_key_id', 'dummykey'), ('Credentials', 'gs_secret_access_key', 'dummysecret')]
        with SetBotoConfigForTest(boto_config_hmac_auth_only):
            bucket_uri = 'gs://any-bucket-name'
            stderr = self.RunGsUtil(self._set_autoclass_cmd + ['on', bucket_uri], return_stderr=True, expected_status=1)
            self.assertIn('command can only be with the Cloud Storage JSON API', stderr)
            stderr = self.RunGsUtil(self._get_autoclass_cmd + [bucket_uri], return_stderr=True, expected_status=1)
            self.assertIn('command can only be with the Cloud Storage JSON API', stderr)

    @SkipForGS('Testing S3 only behavior')
    def test_s3_fails(self):
        bucket_uri = self.CreateBucket()
        stderr = self.RunGsUtil(self._set_autoclass_cmd + ['on', suri(bucket_uri)], return_stderr=True, expected_status=1)
        self.assertIn('command can only be used for GCS Buckets', stderr)
        stderr = self.RunGsUtil(self._get_autoclass_cmd + [suri(bucket_uri)], return_stderr=True, expected_status=1)
        self.assertIn('command can only be used for GCS Buckets', stderr)