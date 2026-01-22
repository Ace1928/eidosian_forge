from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
from random import randint
from unittest import mock
from gslib.cloud_api import AccessDeniedException
from gslib.project_id import PopulateProjectId
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForJSON
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.testcase.integration_testcase import SkipForXML
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import SetEnvironmentForTest
from gslib.utils.retry_util import Retry
from gslib.utils import shim_util
class TestKmsUnitTests(testcase.GsUtilUnitTestCase):
    """Unit tests for gsutil kms."""
    dummy_keyname = 'projects/my-project/locations/us-central1/keyRings/my-keyring/cryptoKeys/my-key'

    @mock.patch('gslib.cloud_api_delegator.CloudApiDelegator.GetProjectServiceAccount')
    @mock.patch('gslib.cloud_api_delegator.CloudApiDelegator.PatchBucket')
    @mock.patch('gslib.kms_api.KmsApi.GetKeyIamPolicy')
    @mock.patch('gslib.kms_api.KmsApi.SetKeyIamPolicy')
    def testEncryptionSetKeySucceedsWhenUpdateKeyPolicySucceeds(self, mock_set_key_iam_policy, mock_get_key_iam_policy, mock_patch_bucket, mock_get_project_service_account):
        bucket_uri = self.CreateBucket()
        mock_get_key_iam_policy.return_value.bindings = []
        mock_get_project_service_account.return_value.email_address = 'dummy@google.com'
        stdout = self.RunCommand('kms', ['encryption', '-k', self.dummy_keyname, suri(bucket_uri)], return_stdout=True)
        self.assertIn('Setting default KMS key for bucket', stdout)
        self.assertTrue(mock_patch_bucket.called)

    @mock.patch('gslib.cloud_api_delegator.CloudApiDelegator.GetProjectServiceAccount')
    @mock.patch('gslib.cloud_api_delegator.CloudApiDelegator.PatchBucket')
    @mock.patch('gslib.kms_api.KmsApi.GetKeyIamPolicy')
    @mock.patch('gslib.kms_api.KmsApi.SetKeyIamPolicy')
    def testEncryptionSetKeySucceedsWhenUpdateKeyPolicyFailsWithWarningFlag(self, mock_set_key_iam_policy, mock_get_key_iam_policy, mock_patch_bucket, mock_get_project_service_account):
        bucket_uri = self.CreateBucket()
        mock_get_key_iam_policy.side_effect = AccessDeniedException('Permission denied')
        mock_get_project_service_account.return_value.email_address = 'dummy@google.com'
        stdout = self.RunCommand('kms', ['encryption', '-k', self.dummy_keyname, '-w', suri(bucket_uri)], return_stdout=True)
        self.assertIn('Setting default KMS key for bucket', stdout)
        self.assertTrue(mock_patch_bucket.called)

    @mock.patch('gslib.cloud_api_delegator.CloudApiDelegator.GetProjectServiceAccount')
    @mock.patch('gslib.cloud_api_delegator.CloudApiDelegator.PatchBucket')
    @mock.patch('gslib.kms_api.KmsApi.GetKeyIamPolicy')
    @mock.patch('gslib.kms_api.KmsApi.SetKeyIamPolicy')
    def testEncryptionSetKeyFailsWhenUpdateKeyPolicyFailsWithoutWarningFlag(self, mock_set_key_iam_policy, mock_get_key_iam_policy, mock_patch_bucket, mock_get_project_service_account):
        bucket_uri = self.CreateBucket()
        mock_get_key_iam_policy.side_effect = AccessDeniedException('Permission denied')
        mock_get_project_service_account.return_value.email_address = 'dummy@google.com'
        try:
            stdout = self.RunCommand('kms', ['encryption', '-k', self.dummy_keyname, suri(bucket_uri)], return_stdout=True)
            self.fail('Did not get expected AccessDeniedException')
        except AccessDeniedException as e:
            self.assertIn('Permission denied', e.reason)

    @mock.patch('gslib.cloud_api_delegator.CloudApiDelegator.GetProjectServiceAccount')
    @mock.patch('gslib.kms_api.KmsApi.GetKeyIamPolicy')
    @mock.patch('gslib.kms_api.KmsApi.SetKeyIamPolicy')
    def test_shim_translates_authorize_flags(self, mock_get_key_iam_policy, mock_set_key_iam_policy, mock_get_project_service_account):
        del mock_set_key_iam_policy
        mock_get_project_service_account.return_value.email_address = 'dummy@google.com'
        mock_get_key_iam_policy.return_value.bindings = []
        with SetBotoConfigForTest([('GSUtil', 'use_gcloud_storage', 'True'), ('GSUtil', 'hidden_shim_mode', 'dry_run')]):
            with SetEnvironmentForTest({'CLOUDSDK_CORE_PASS_CREDENTIALS_TO_GSUTIL': 'True', 'CLOUDSDK_ROOT_DIR': 'fake_dir'}):
                mock_log_handler = self.RunCommand('kms', ['authorize', '-p', 'foo', '-k', self.dummy_keyname], return_log_handler=True)
                info_lines = '\n'.join(mock_log_handler.messages['info'])
                self.assertIn('Gcloud Storage Command: {} storage service-agent --project foo --authorize-cmek {}'.format(shim_util._get_gcloud_binary_path('fake_dir'), self.dummy_keyname), info_lines)

    def test_shim_translates_clear_encryption_key(self):
        bucket_uri = self.CreateBucket()
        with SetBotoConfigForTest([('GSUtil', 'use_gcloud_storage', 'True'), ('GSUtil', 'hidden_shim_mode', 'dry_run')]):
            with SetEnvironmentForTest({'CLOUDSDK_CORE_PASS_CREDENTIALS_TO_GSUTIL': 'True', 'CLOUDSDK_ROOT_DIR': 'fake_dir'}):
                mock_log_handler = self.RunCommand('kms', ['encryption', '-d', suri(bucket_uri)], return_log_handler=True)
                info_lines = '\n'.join(mock_log_handler.messages['info'])
                self.assertIn('Gcloud Storage Command: {} storage buckets update --clear-default-encryption-key {}'.format(shim_util._get_gcloud_binary_path('fake_dir'), suri(bucket_uri)), info_lines)

    @mock.patch('gslib.cloud_api_delegator.CloudApiDelegator.GetProjectServiceAccount')
    @mock.patch('gslib.kms_api.KmsApi.GetKeyIamPolicy')
    @mock.patch('gslib.kms_api.KmsApi.SetKeyIamPolicy')
    def test_shim_translates_update_encryption_key(self, mock_get_key_iam_policy, mock_set_key_iam_policy, mock_get_project_service_account):
        bucket_uri = self.CreateBucket()
        del mock_set_key_iam_policy
        mock_get_project_service_account.return_value.email_address = 'dummy@google.com'
        mock_get_key_iam_policy.return_value.bindings = []
        with SetBotoConfigForTest([('GSUtil', 'use_gcloud_storage', 'True'), ('GSUtil', 'hidden_shim_mode', 'dry_run')]):
            with SetEnvironmentForTest({'CLOUDSDK_CORE_PASS_CREDENTIALS_TO_GSUTIL': 'True', 'CLOUDSDK_ROOT_DIR': 'fake_dir'}):
                mock_log_handler = self.RunCommand('kms', ['encryption', '-w', '-k', self.dummy_keyname, suri(bucket_uri)], return_log_handler=True)
                info_lines = '\n'.join(mock_log_handler.messages['info'])
                self.assertIn('Gcloud Storage Command: {} storage buckets update  --default-encryption-key {} {}'.format(shim_util._get_gcloud_binary_path('fake_dir'), self.dummy_keyname, suri(bucket_uri)), info_lines)

    def test_shim_translates_displays_encryption_key(self):
        bucket_uri = self.CreateBucket()
        with SetBotoConfigForTest([('GSUtil', 'use_gcloud_storage', 'True'), ('GSUtil', 'hidden_shim_mode', 'dry_run')]):
            with SetEnvironmentForTest({'CLOUDSDK_CORE_PASS_CREDENTIALS_TO_GSUTIL': 'True', 'CLOUDSDK_ROOT_DIR': 'fake_dir'}):
                mock_log_handler = self.RunCommand('kms', ['encryption', suri(bucket_uri)], return_log_handler=True)
                info_lines = '\n'.join(mock_log_handler.messages['info'])
                self.assertIn('Gcloud Storage Command: {} storage buckets describe --format=value[separator=": "](name, encryption.defaultKmsKeyName.yesno(no="No default encryption key.")) --raw {}'.format(shim_util._get_gcloud_binary_path('fake_dir'), suri(bucket_uri)), info_lines)

    @mock.patch('gslib.cloud_api_delegator.CloudApiDelegator.GetProjectServiceAccount')
    def test_shim_translates_serviceaccount_command(self, mock_get_project_service_account):
        mock_get_project_service_account.return_value.email_address = 'dummy@google.com'
        with SetBotoConfigForTest([('GSUtil', 'use_gcloud_storage', 'True'), ('GSUtil', 'hidden_shim_mode', 'dry_run')]):
            with SetEnvironmentForTest({'CLOUDSDK_CORE_PASS_CREDENTIALS_TO_GSUTIL': 'True', 'CLOUDSDK_ROOT_DIR': 'fake_dir'}):
                mock_log_handler = self.RunCommand('kms', ['serviceaccount', '-p', 'foo'], return_log_handler=True)
                info_lines = '\n'.join(mock_log_handler.messages['info'])
                self.assertIn('Gcloud Storage Command: {} storage service-agent --project foo'.format(shim_util._get_gcloud_binary_path('fake_dir')), info_lines)