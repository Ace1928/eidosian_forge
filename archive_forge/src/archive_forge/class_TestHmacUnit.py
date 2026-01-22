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
class TestHmacUnit(testcase.GsUtilUnitTestCase):

    @mock.patch.object(hmac.HmacCommand, 'RunCommand', new=mock.Mock())
    def test_shim_translates_hmac_create_command(self):
        fake_cloudsdk_dir = 'fake_dir'
        service_account = 'test.service.account@test_project.iam.gserviceaccount.com'
        project = 'test_project'
        with SetBotoConfigForTest([('GSUtil', 'use_gcloud_storage', 'True'), ('GSUtil', 'hidden_shim_mode', 'dry_run')]):
            with SetEnvironmentForTest({'CLOUDSDK_CORE_PASS_CREDENTIALS_TO_GSUTIL': 'True', 'CLOUDSDK_ROOT_DIR': fake_cloudsdk_dir}):
                mock_log_handler = self.RunCommand('hmac', args=['create', '-p', project, service_account], return_log_handler=True)
                info_lines = '\n'.join(mock_log_handler.messages['info'])
                self.assertIn('Gcloud Storage Command: {} storage hmac create {} --project {} {}'.format(shim_util._get_gcloud_binary_path('fake_dir'), hmac._CREATE_COMMAND_FORMAT, project, service_account), info_lines)

    @mock.patch.object(hmac.HmacCommand, 'RunCommand', new=mock.Mock())
    def test_shim_translates_delete_command(self):
        fake_cloudsdk_dir = 'fake_dir'
        project = 'test-project'
        access_id = 'fake123456789'
        with SetBotoConfigForTest([('GSUtil', 'use_gcloud_storage', 'True'), ('GSUtil', 'hidden_shim_mode', 'dry_run')]):
            with SetEnvironmentForTest({'CLOUDSDK_CORE_PASS_CREDENTIALS_TO_GSUTIL': 'True', 'CLOUDSDK_ROOT_DIR': fake_cloudsdk_dir}):
                mock_log_handler = self.RunCommand('hmac', args=['delete', '-p', project, access_id], return_log_handler=True)
                info_lines = '\n'.join(mock_log_handler.messages['info'])
                self.assertIn('Gcloud Storage Command: {} storage hmac delete --project {} {}'.format(shim_util._get_gcloud_binary_path('fake_dir'), project, access_id), info_lines)

    @mock.patch.object(hmac.HmacCommand, 'RunCommand', new=mock.Mock())
    def test_shim_translates_get_commannd(self):
        fake_cloudsdk_dir = 'fake_dir'
        project = 'test-project'
        access_id = 'fake123456789'
        with SetBotoConfigForTest([('GSUtil', 'use_gcloud_storage', 'True'), ('GSUtil', 'hidden_shim_mode', 'dry_run')]):
            with SetEnvironmentForTest({'CLOUDSDK_CORE_PASS_CREDENTIALS_TO_GSUTIL': 'True', 'CLOUDSDK_ROOT_DIR': fake_cloudsdk_dir}):
                mock_log_handler = self.RunCommand('hmac', args=['get', '-p', project, access_id], return_log_handler=True)
                info_lines = '\n'.join(mock_log_handler.messages['info'])
                self.assertIn('Gcloud Storage Command: {} storage hmac describe {} --project {} {}'.format(shim_util._get_gcloud_binary_path('fake_dir'), hmac._DESCRIBE_COMMAND_FORMAT, project, access_id), info_lines)

    @mock.patch.object(hmac.HmacCommand, 'RunCommand', new=mock.Mock())
    def test_shim_translates_hmac_list_command_using_short_format(self):
        fake_cloudsdk_dir = 'fake_dir'
        project = 'test-project'
        service_account = 'test.service.account@test_project.iam.gserviceaccount.com'
        with SetBotoConfigForTest([('GSUtil', 'use_gcloud_storage', 'True'), ('GSUtil', 'hidden_shim_mode', 'dry_run')]):
            with SetEnvironmentForTest({'CLOUDSDK_CORE_PASS_CREDENTIALS_TO_GSUTIL': 'True', 'CLOUDSDK_ROOT_DIR': fake_cloudsdk_dir}):
                mock_log_handler = self.RunCommand('hmac', args=['list', '-a', '-u', service_account, '-p', project], return_log_handler=True)
                info_lines = '\n'.join(mock_log_handler.messages['info'])
                self.assertIn('Gcloud Storage Command: {} storage hmac list {} --all --service-account {} --project {}'.format(shim_util._get_gcloud_binary_path('fake_dir'), hmac._LIST_COMMAND_SHORT_FORMAT, service_account, project), info_lines)

    @mock.patch.object(hmac.HmacCommand, 'RunCommand', new=mock.Mock())
    def test_shim_translates_hmac_list_command_using_long_format(self):
        fake_cloudsdk_dir = 'fake_dir'
        project = 'test-project'
        service_account = 'test.service.account@test_project.iam.gserviceaccount.com'
        with SetBotoConfigForTest([('GSUtil', 'use_gcloud_storage', 'True'), ('GSUtil', 'hidden_shim_mode', 'dry_run')]):
            with SetEnvironmentForTest({'CLOUDSDK_CORE_PASS_CREDENTIALS_TO_GSUTIL': 'True', 'CLOUDSDK_ROOT_DIR': fake_cloudsdk_dir}):
                mock_log_handler = self.RunCommand('hmac', args=['list', '-a', '-u', service_account, '-l', '-p', project], return_log_handler=True)
                info_lines = '\n'.join(mock_log_handler.messages['info'])
                self.assertIn('Gcloud Storage Command: {} storage hmac list {} --all --service-account {} --long --project {}'.format(shim_util._get_gcloud_binary_path('fake_dir'), hmac._DESCRIBE_COMMAND_FORMAT, service_account, project), info_lines)

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

    @mock.patch.object(hmac.HmacCommand, 'RunCommand', new=mock.Mock())
    def test_shim_translates_hmac_update_command_when_inactive_state_option_is_passed(self):
        fake_cloudsdk_dir = 'fake_dir'
        etag = 'ABCDEFGHIK='
        project = 'test-project'
        access_id = 'fake123456789'
        with SetBotoConfigForTest([('GSUtil', 'use_gcloud_storage', 'True'), ('GSUtil', 'hidden_shim_mode', 'dry_run')]):
            with SetEnvironmentForTest({'CLOUDSDK_CORE_PASS_CREDENTIALS_TO_GSUTIL': 'True', 'CLOUDSDK_ROOT_DIR': fake_cloudsdk_dir}):
                mock_log_handler = self.RunCommand('hmac', args=['update', '-e', etag, '-p', project, '-s', 'INACTIVE', access_id], return_log_handler=True)
                info_lines = '\n'.join(mock_log_handler.messages['info'])
                self.assertIn('Gcloud Storage Command: {} storage hmac update {} --etag {} --project {} --{} {}'.format(shim_util._get_gcloud_binary_path('fake_dir'), hmac._DESCRIBE_COMMAND_FORMAT, etag, project, 'deactivate', access_id), info_lines)