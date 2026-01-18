from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import boto
import os
import re
from gslib.commands import hmac
from gslib.project_id import PopulateProjectId
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.testcase.integration_testcase import SkipForXML
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import SetEnvironmentForTest
from gslib.tests.util import unittest
from gslib.utils.retry_util import Retry
from gslib.utils import shim_util
from six import add_move, MovedModule
from six.moves import mock
@mock.patch.object(hmac.HmacCommand, 'RunCommand', new=mock.Mock())
def test_shim_translates_hmac_update_command_when_active_state_option_is_passed(self):
    fake_cloudsdk_dir = 'fake_dir'
    etag = 'ABCDEFGHIK='
    project = 'test-project'
    access_id = 'fake123456789'
    with SetBotoConfigForTest([('GSUtil', 'use_gcloud_storage', 'True'), ('GSUtil', 'hidden_shim_mode', 'dry_run')]):
        with SetEnvironmentForTest({'CLOUDSDK_CORE_PASS_CREDENTIALS_TO_GSUTIL': 'True', 'CLOUDSDK_ROOT_DIR': fake_cloudsdk_dir}):
            mock_log_handler = self.RunCommand('hmac', args=['update', '-e', etag, '-p', project, '-s', 'ACTIVE', access_id], return_log_handler=True)
            info_lines = '\n'.join(mock_log_handler.messages['info'])
            self.assertIn('Gcloud Storage Command: {} storage hmac update {} --etag {} --project {} --{} {}'.format(shim_util._get_gcloud_binary_path('fake_dir'), hmac._DESCRIBE_COMMAND_FORMAT, etag, project, 'activate', access_id), info_lines)