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