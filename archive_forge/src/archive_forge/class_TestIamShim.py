from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from collections import defaultdict
import json
import os
import subprocess
from gslib.commands import iam
from gslib.exception import CommandException
from gslib.project_id import PopulateProjectId
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.testcase.integration_testcase import SkipForXML
from gslib.tests.util import GenerationFromURI as urigen
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import SetEnvironmentForTest
from gslib.tests.util import unittest
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
from gslib.utils import shim_util
from gslib.utils.constants import UTF8
from gslib.utils.iam_helper import BindingsMessageToUpdateDict
from gslib.utils.iam_helper import BindingsDictToUpdateDict
from gslib.utils.iam_helper import BindingStringToTuple as bstt
from gslib.utils.iam_helper import DiffBindings
from gslib.utils.iam_helper import IsEqualBindings
from gslib.utils.iam_helper import PatchBindings
from gslib.utils.retry_util import Retry
from six import add_move, MovedModule
from six.moves import mock
class TestIamShim(testcase.GsUtilUnitTestCase):

    @mock.patch.object(iam.IamCommand, 'RunCommand', new=mock.Mock())
    def test_shim_translates_iam_get_object(self):
        with SetBotoConfigForTest([('GSUtil', 'use_gcloud_storage', 'True'), ('GSUtil', 'hidden_shim_mode', 'dry_run')]):
            with SetEnvironmentForTest({'CLOUDSDK_CORE_PASS_CREDENTIALS_TO_GSUTIL': 'True', 'CLOUDSDK_ROOT_DIR': 'fake_dir'}):
                mock_log_handler = self.RunCommand('iam', ['get', 'gs://bucket/object'], return_log_handler=True)
                info_lines = '\n'.join(mock_log_handler.messages['info'])
                self.assertIn('Gcloud Storage Command: {} storage objects get-iam-policy --format=json gs://bucket/object'.format(shim_util._get_gcloud_binary_path('fake_dir')), info_lines)

    @mock.patch.object(iam.IamCommand, 'RunCommand', new=mock.Mock())
    def test_shim_translates_iam_get_bucket(self):
        with SetBotoConfigForTest([('GSUtil', 'use_gcloud_storage', 'True'), ('GSUtil', 'hidden_shim_mode', 'dry_run')]):
            with SetEnvironmentForTest({'CLOUDSDK_CORE_PASS_CREDENTIALS_TO_GSUTIL': 'True', 'CLOUDSDK_ROOT_DIR': 'fake_dir'}):
                mock_log_handler = self.RunCommand('iam', ['get', 'gs://bucket'], return_log_handler=True)
                info_lines = '\n'.join(mock_log_handler.messages['info'])
                self.assertIn('Gcloud Storage Command: {} storage buckets get-iam-policy --format=json gs://bucket'.format(shim_util._get_gcloud_binary_path('fake_dir')), info_lines)

    @mock.patch.object(iam.IamCommand, 'RunCommand', new=mock.Mock())
    def test_shim_translates_iam_set_object(self):
        with SetBotoConfigForTest([('GSUtil', 'use_gcloud_storage', 'True'), ('GSUtil', 'hidden_shim_mode', 'dry_run')]):
            with SetEnvironmentForTest({'CLOUDSDK_CORE_PASS_CREDENTIALS_TO_GSUTIL': 'True', 'CLOUDSDK_ROOT_DIR': 'fake_dir'}):
                mock_log_handler = self.RunCommand('iam', ['set', 'policy-file', 'gs://b/o1', 'gs://b/o2'], return_log_handler=True)
                info_lines = '\n'.join(mock_log_handler.messages['info'])
                self.assertIn('Gcloud Storage Command: {} storage objects set-iam-policy --format=json gs://b/o1 gs://b/o2 policy-file'.format(shim_util._get_gcloud_binary_path('fake_dir')), info_lines)

    @mock.patch.object(iam.IamCommand, 'RunCommand', new=mock.Mock())
    def test_shim_translates_iam_set_bucket(self):
        with SetBotoConfigForTest([('GSUtil', 'use_gcloud_storage', 'True'), ('GSUtil', 'hidden_shim_mode', 'dry_run')]):
            with SetEnvironmentForTest({'CLOUDSDK_CORE_PASS_CREDENTIALS_TO_GSUTIL': 'True', 'CLOUDSDK_ROOT_DIR': 'fake_dir'}):
                mock_log_handler = self.RunCommand('iam', ['set', 'policy-file', 'gs://b1', 'gs://b2'], return_log_handler=True)
                info_lines = '\n'.join(mock_log_handler.messages['info'])
                self.assertIn('Gcloud Storage Command: {} storage buckets set-iam-policy --format=json gs://b1 gs://b2 policy-file'.format(shim_util._get_gcloud_binary_path('fake_dir')), info_lines)

    @mock.patch.object(iam.IamCommand, 'RunCommand', new=mock.Mock())
    def test_shim_translates_iam_set_mix_of_bucket_and_objects_if_recursive(self):
        with SetBotoConfigForTest([('GSUtil', 'use_gcloud_storage', 'True'), ('GSUtil', 'hidden_shim_mode', 'dry_run')]):
            with SetEnvironmentForTest({'CLOUDSDK_CORE_PASS_CREDENTIALS_TO_GSUTIL': 'True', 'CLOUDSDK_ROOT_DIR': 'fake_dir'}):
                mock_log_handler = self.RunCommand('iam', ['set', '-r', 'policy-file', 'gs://b1', 'gs://b2/o'], return_log_handler=True)
                info_lines = '\n'.join(mock_log_handler.messages['info'])
                self.assertIn('Gcloud Storage Command: {} storage objects set-iam-policy --format=json --recursive gs://b1 gs://b2/o policy-file'.format(shim_util._get_gcloud_binary_path('fake_dir')), info_lines)

    @mock.patch.object(iam.IamCommand, 'RunCommand', new=mock.Mock())
    def test_shim_raises_for_iam_set_mix_of_bucket_and_objects(self):
        with SetBotoConfigForTest([('GSUtil', 'use_gcloud_storage', 'True'), ('GSUtil', 'hidden_shim_mode', 'dry_run')]):
            with SetEnvironmentForTest({'CLOUDSDK_CORE_PASS_CREDENTIALS_TO_GSUTIL': 'True', 'CLOUDSDK_ROOT_DIR': 'fake_dir'}):
                with self.assertRaisesRegex(CommandException, 'Cannot operate on a mix of buckets and objects.'):
                    self.RunCommand('iam', ['set', 'policy-file', 'gs://b', 'gs://b/o'])

    @mock.patch.object(iam.IamCommand, 'RunCommand', new=mock.Mock())
    def test_shim_translates_iam_set_handles_valid_etag(self):
        with SetBotoConfigForTest([('GSUtil', 'use_gcloud_storage', 'True'), ('GSUtil', 'hidden_shim_mode', 'dry_run')]):
            with SetEnvironmentForTest({'CLOUDSDK_CORE_PASS_CREDENTIALS_TO_GSUTIL': 'True', 'CLOUDSDK_ROOT_DIR': 'fake_dir'}):
                mock_log_handler = self.RunCommand('iam', ['set', '-e', 'abc=', 'policy-file', 'gs://b'], return_log_handler=True)
                info_lines = '\n'.join(mock_log_handler.messages['info'])
                self.assertIn('Gcloud Storage Command: {} storage buckets set-iam-policy --format=json --etag abc= gs://b policy-file'.format(shim_util._get_gcloud_binary_path('fake_dir')), info_lines)

    @mock.patch.object(iam.IamCommand, 'RunCommand', new=mock.Mock())
    def test_shim_translates_iam_set_handles_empty_etag(self):
        with SetBotoConfigForTest([('GSUtil', 'use_gcloud_storage', 'True'), ('GSUtil', 'hidden_shim_mode', 'dry_run')]):
            with SetEnvironmentForTest({'CLOUDSDK_CORE_PASS_CREDENTIALS_TO_GSUTIL': 'True', 'CLOUDSDK_ROOT_DIR': 'fake_dir'}):
                mock_log_handler = self.RunCommand('iam', ['set', '-e', '', 'policy-file', 'gs://b'], return_log_handler=True)
                info_lines = '\n'.join(mock_log_handler.messages['info'])
                self.assertIn('Gcloud Storage Command: {} storage buckets set-iam-policy --format=json --etag= gs://b policy-file'.format(shim_util._get_gcloud_binary_path('fake_dir')), info_lines)

    @mock.patch.object(iam.IamCommand, 'RunCommand', new=mock.Mock())
    def test_shim_warns_with_dry_run_mode_for_iam_ch(self):
        with SetBotoConfigForTest([('GSUtil', 'use_gcloud_storage', 'True'), ('GSUtil', 'hidden_shim_mode', 'dry_run')]):
            with SetEnvironmentForTest({'CLOUDSDK_CORE_PASS_CREDENTIALS_TO_GSUTIL': 'True', 'CLOUDSDK_ROOT_DIR': 'fake_dir'}):
                mock_log_handler = self.RunCommand('iam', ['ch', '-d', 'allUsers', 'gs://b'], return_log_handler=True)
                warning_lines = '\n'.join(mock_log_handler.messages['warning'])
                self.assertIn('The shim maps iam ch commands to several gcloud storage commands, which cannot be determined without running gcloud storage.', warning_lines)

    def _get_run_call(self, command, env=mock.ANY, stdin=None, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True):
        return mock.call(command, env=env, input=stdin, stderr=stderr, stdout=stdout, text=text)

    @mock.patch.object(subprocess, 'run', autospec=True)
    def test_iam_ch_adds_updates_and_deletes_bucket_policies(self, mock_run):
        original_policy = {'bindings': [{'role': 'preserved-role', 'members': ['allUsers']}, {'role': 'roles/storage.modified-role', 'members': ['allUsers', 'user:deleted-user@example.com']}, {'role': 'roles/storage.deleted-role', 'members': ['allUsers']}]}
        new_policy = {'bindings': [{'role': 'preserved-role', 'members': ['allUsers']}, {'role': 'roles/storage.modified-role', 'members': ['allAuthenticatedUsers', 'allUsers']}]}
        with SetBotoConfigForTest([('GSUtil', 'use_gcloud_storage', 'True')]):
            get_process = subprocess.CompletedProcess(args=[], returncode=0, stdout=json.dumps(original_policy))
            set_process = subprocess.CompletedProcess(args=[], returncode=0)
            mock_run.side_effect = [get_process, set_process]
            with SetEnvironmentForTest({'CLOUDSDK_CORE_PASS_CREDENTIALS_TO_GSUTIL': 'True', 'CLOUDSDK_ROOT_DIR': 'fake_dir'}):
                self.RunCommand('iam', ['ch', 'allAuthenticatedUsers:modified-role', '-d', 'user:deleted-user@example.com', '-d', 'allUsers:deleted-role', 'gs://b'])
            self.assertEqual(mock_run.call_args_list, [self._get_run_call([shim_util._get_gcloud_binary_path('fake_dir'), 'storage', 'buckets', 'get-iam-policy', 'gs://b/', '--format=json']), self._get_run_call([shim_util._get_gcloud_binary_path('fake_dir'), 'storage', 'buckets', 'set-iam-policy', 'gs://b/', '-'], stdin=json.dumps(new_policy, sort_keys=True))])

    @mock.patch.object(subprocess, 'run', autospec=True)
    def test_iam_ch_updates_bucket_policies_for_multiple_urls(self, mock_run):
        original_policy1 = {'bindings': [{'role': 'roles/storage.modified-role', 'members': ['user:test-user1@example.com']}]}
        original_policy2 = {'bindings': [{'role': 'roles/storage.modified-role', 'members': ['user:test-user2@example.com']}]}
        new_policy1 = {'bindings': [{'role': 'roles/storage.modified-role', 'members': ['allAuthenticatedUsers', 'user:test-user1@example.com']}]}
        new_policy2 = {'bindings': [{'role': 'roles/storage.modified-role', 'members': ['allAuthenticatedUsers', 'user:test-user2@example.com']}]}
        with SetBotoConfigForTest([('GSUtil', 'use_gcloud_storage', 'True')]):
            get_process1 = subprocess.CompletedProcess(args=[], returncode=0, stdout=json.dumps(original_policy1))
            get_process2 = subprocess.CompletedProcess(args=[], returncode=0, stdout=json.dumps(original_policy2))
            set_process = subprocess.CompletedProcess(args=[], returncode=0)
            mock_run.side_effect = [get_process1, set_process, get_process2, set_process]
            with SetEnvironmentForTest({'CLOUDSDK_CORE_PASS_CREDENTIALS_TO_GSUTIL': 'True', 'CLOUDSDK_ROOT_DIR': 'fake_dir'}):
                self.RunCommand('iam', ['ch', 'allAuthenticatedUsers:modified-role', 'gs://b1', 'gs://b2'])
            self.assertEqual(mock_run.call_args_list, [self._get_run_call([shim_util._get_gcloud_binary_path('fake_dir'), 'storage', 'buckets', 'get-iam-policy', 'gs://b1/', '--format=json']), self._get_run_call([shim_util._get_gcloud_binary_path('fake_dir'), 'storage', 'buckets', 'set-iam-policy', 'gs://b1/', '-'], stdin=json.dumps(new_policy1, sort_keys=True)), self._get_run_call([shim_util._get_gcloud_binary_path('fake_dir'), 'storage', 'buckets', 'get-iam-policy', 'gs://b2/', '--format=json']), self._get_run_call([shim_util._get_gcloud_binary_path('fake_dir'), 'storage', 'buckets', 'set-iam-policy', 'gs://b2/', '-'], stdin=json.dumps(new_policy2, sort_keys=True))])

    @mock.patch.object(subprocess, 'run', autospec=True)
    def test_iam_ch_updates_object_policies(self, mock_run):
        original_policy = {'bindings': [{'role': 'roles/storage.modified-role', 'members': ['allUsers']}]}
        new_policy = {'bindings': [{'role': 'roles/storage.modified-role', 'members': ['allAuthenticatedUsers', 'allUsers']}]}
        with SetBotoConfigForTest([('GSUtil', 'use_gcloud_storage', 'True')]):
            ls_process = subprocess.CompletedProcess(args=[], returncode=0, stdout=json.dumps([{'url': 'gs://b/o', 'type': 'cloud_object'}]))
            get_process = subprocess.CompletedProcess(args=[], returncode=0, stdout=json.dumps(original_policy))
            set_process = subprocess.CompletedProcess(args=[], returncode=0)
            mock_run.side_effect = [ls_process, get_process, set_process]
            with SetEnvironmentForTest({'CLOUDSDK_CORE_PASS_CREDENTIALS_TO_GSUTIL': 'True', 'CLOUDSDK_ROOT_DIR': 'fake_dir'}):
                self.RunCommand('iam', ['ch', 'allAuthenticatedUsers:modified-role', 'gs://b/o'])
            self.assertEqual(mock_run.call_args_list, [self._get_run_call([shim_util._get_gcloud_binary_path('fake_dir'), 'storage', 'ls', '--json', 'gs://b/o']), self._get_run_call([shim_util._get_gcloud_binary_path('fake_dir'), 'storage', 'objects', 'get-iam-policy', 'gs://b/o', '--format=json']), self._get_run_call([shim_util._get_gcloud_binary_path('fake_dir'), 'storage', 'objects', 'set-iam-policy', 'gs://b/o', '-'], stdin=json.dumps(new_policy, sort_keys=True))])

    @mock.patch.object(subprocess, 'run', autospec=True)
    def test_iam_ch_expands_urls_with_recursion_and_ignores_container_headers(self, mock_run):
        original_policy = {'bindings': [{'role': 'modified-role', 'members': ['allUsers']}]}
        with SetBotoConfigForTest([('GSUtil', 'use_gcloud_storage', 'True')]):
            ls_process = subprocess.CompletedProcess(args=[], returncode=0, stdout=json.dumps([{'url': 'gs://b/dir/', 'type': 'prefix'}, {'url': 'gs://b/dir/:', 'type': 'cloud_object'}, {'url': 'gs://b/dir2/', 'type': 'prefix'}, {'url': 'gs://b/dir2/o', 'type': 'cloud_object'}]))
            get_process = subprocess.CompletedProcess(args=[], returncode=0, stdout=json.dumps(original_policy))
            set_process = subprocess.CompletedProcess(args=[], returncode=0)
            mock_run.side_effect = [ls_process] + [get_process, set_process] * 3
            with SetEnvironmentForTest({'CLOUDSDK_CORE_PASS_CREDENTIALS_TO_GSUTIL': 'True', 'CLOUDSDK_ROOT_DIR': 'fake_dir'}):
                self.RunCommand('iam', ['ch', '-r', 'allAuthenticatedUsers:modified-role', 'gs://b'])
            self.assertEqual(mock_run.call_args_list, [self._get_run_call([shim_util._get_gcloud_binary_path('fake_dir'), 'storage', 'ls', '--json', '-r', 'gs://b/']), self._get_run_call([shim_util._get_gcloud_binary_path('fake_dir'), 'storage', 'objects', 'get-iam-policy', 'gs://b/dir/:', '--format=json']), self._get_run_call([shim_util._get_gcloud_binary_path('fake_dir'), 'storage', 'objects', 'set-iam-policy', 'gs://b/dir/:', '-'], stdin=mock.ANY), self._get_run_call([shim_util._get_gcloud_binary_path('fake_dir'), 'storage', 'objects', 'get-iam-policy', 'gs://b/dir2/o', '--format=json']), self._get_run_call([shim_util._get_gcloud_binary_path('fake_dir'), 'storage', 'objects', 'set-iam-policy', 'gs://b/dir2/o', '-'], stdin=mock.ANY)])

    @mock.patch.object(subprocess, 'run', autospec=True)
    def test_iam_ch_raises_ls_error(self, mock_run):
        with SetBotoConfigForTest([('GSUtil', 'use_gcloud_storage', 'True')]):
            ls_process = subprocess.CompletedProcess(args=[], returncode=1, stderr='An error.')
            mock_run.side_effect = [ls_process]
            with SetEnvironmentForTest({'CLOUDSDK_CORE_PASS_CREDENTIALS_TO_GSUTIL': 'True', 'CLOUDSDK_ROOT_DIR': 'fake_dir'}):
                with self.assertRaisesRegex(CommandException, 'An error.'):
                    self.RunCommand('iam', ['ch', 'allAuthenticatedUsers:modified-role', 'gs://b/o'])
                self.assertEqual(mock_run.call_count, 1)

    @mock.patch.object(subprocess, 'run', autospec=True)
    def test_iam_ch_raises_get_error(self, mock_run):
        with SetBotoConfigForTest([('GSUtil', 'use_gcloud_storage', 'True')]):
            ls_process = subprocess.CompletedProcess(args=[], returncode=0, stdout=json.dumps([{'url': 'gs://b/o', 'type': 'cloud_object'}]))
            get_process = subprocess.CompletedProcess(args=[], returncode=1, stderr='An error.')
            mock_run.side_effect = [ls_process, get_process]
            with SetEnvironmentForTest({'CLOUDSDK_CORE_PASS_CREDENTIALS_TO_GSUTIL': 'True', 'CLOUDSDK_ROOT_DIR': 'fake_dir'}):
                with self.assertRaisesRegex(CommandException, 'An error.'):
                    self.RunCommand('iam', ['ch', 'allAuthenticatedUsers:modified-role', 'gs://b/o'])
                self.assertEqual(mock_run.call_count, 2)

    @mock.patch.object(subprocess, 'run', autospec=True)
    def test_iam_ch_raises_set_error(self, mock_run):
        with SetBotoConfigForTest([('GSUtil', 'use_gcloud_storage', 'True')]):
            ls_process = subprocess.CompletedProcess(args=[], returncode=0, stdout=json.dumps([{'url': 'gs://b/o', 'type': 'cloud_object'}]))
            get_process = subprocess.CompletedProcess(args=[], returncode=0, stdout='{"bindings": []}')
            set_process = subprocess.CompletedProcess(args=[], returncode=1, stderr='An error.')
            mock_run.side_effect = [ls_process, get_process, set_process]
            with SetEnvironmentForTest({'CLOUDSDK_CORE_PASS_CREDENTIALS_TO_GSUTIL': 'True', 'CLOUDSDK_ROOT_DIR': 'fake_dir'}):
                with self.assertRaisesRegex(CommandException, 'An error.'):
                    self.RunCommand('iam', ['ch', 'allAuthenticatedUsers:modified-role', 'gs://b/o'])
                self.assertEqual(mock_run.call_count, 3)

    @mock.patch.object(subprocess, 'run', autospec=True)
    def test_iam_ch_continues_on_ls_error(self, mock_run):
        original_policy = {'bindings': [{'role': 'roles/storage.modified-role', 'members': ['allUsers']}]}
        new_policy = {'bindings': [{'role': 'roles/storage.modified-role', 'members': ['allAuthenticatedUsers', 'allUsers']}]}
        with SetBotoConfigForTest([('GSUtil', 'use_gcloud_storage', 'True')]):
            ls_process = subprocess.CompletedProcess(args=[], returncode=1, stderr='An error.')
            ls_process2 = subprocess.CompletedProcess(args=[], returncode=1, stderr='Another error.')
            mock_run.side_effect = [ls_process, ls_process2]
            with SetEnvironmentForTest({'CLOUDSDK_CORE_PASS_CREDENTIALS_TO_GSUTIL': 'True', 'CLOUDSDK_ROOT_DIR': 'fake_dir'}):
                mock_log_handler = self.RunCommand('iam', ['ch', '-f', 'allAuthenticatedUsers:modified-role', 'gs://b/o1', 'gs://b/o2'], debug=1, return_log_handler=True)
            self.assertEqual(mock_run.call_args_list, [self._get_run_call([shim_util._get_gcloud_binary_path('fake_dir'), 'storage', 'ls', '--json', 'gs://b/o1']), self._get_run_call([shim_util._get_gcloud_binary_path('fake_dir'), 'storage', 'ls', '--json', 'gs://b/o2'])])
            error_lines = '\n'.join(mock_log_handler.messages['error'])
            self.assertIn('An error.', error_lines)
            self.assertIn('Another error.', error_lines)

    @mock.patch.object(subprocess, 'run', autospec=True)
    def test_iam_ch_continues_on_get_error(self, mock_run):
        original_policy = {'bindings': [{'role': 'roles/storage.modified-role', 'members': ['allUsers']}]}
        new_policy = {'bindings': [{'role': 'roles/storage.modified-role', 'members': ['allAuthenticatedUsers', 'allUsers']}]}
        with SetBotoConfigForTest([('GSUtil', 'use_gcloud_storage', 'True')]):
            ls_process = subprocess.CompletedProcess(args=[], returncode=0, stdout=json.dumps([{'url': 'gs://b/o1', 'type': 'cloud_object'}]))
            get_process = subprocess.CompletedProcess(args=[], returncode=1, stderr='An error.')
            ls_process2 = subprocess.CompletedProcess(args=[], returncode=1, stderr='Another error.')
            mock_run.side_effect = [ls_process, get_process, ls_process2]
            with SetEnvironmentForTest({'CLOUDSDK_CORE_PASS_CREDENTIALS_TO_GSUTIL': 'True', 'CLOUDSDK_ROOT_DIR': 'fake_dir'}):
                mock_log_handler = self.RunCommand('iam', ['ch', '-f', 'allAuthenticatedUsers:modified-role', 'gs://b/o1', 'gs://b/o2'], debug=1, return_log_handler=True)
            self.assertEqual(mock_run.call_args_list, [self._get_run_call([shim_util._get_gcloud_binary_path('fake_dir'), 'storage', 'ls', '--json', 'gs://b/o1']), self._get_run_call([shim_util._get_gcloud_binary_path('fake_dir'), 'storage', 'objects', 'get-iam-policy', 'gs://b/o1', '--format=json']), self._get_run_call([shim_util._get_gcloud_binary_path('fake_dir'), 'storage', 'ls', '--json', 'gs://b/o2'])])
            error_lines = '\n'.join(mock_log_handler.messages['error'])
            self.assertIn('An error.', error_lines)
            self.assertIn('Another error.', error_lines)

    @mock.patch.object(subprocess, 'run', autospec=True)
    def test_iam_ch_continues_on_set_error(self, mock_run):
        original_policy = {'bindings': [{'role': 'roles/storage.modified-role', 'members': ['allUsers']}]}
        new_policy = {'bindings': [{'role': 'roles/storage.modified-role', 'members': ['allAuthenticatedUsers', 'allUsers']}]}
        with SetBotoConfigForTest([('GSUtil', 'use_gcloud_storage', 'True')]):
            ls_process = subprocess.CompletedProcess(args=[], returncode=0, stdout=json.dumps([{'url': 'gs://b/o1', 'type': 'cloud_object'}]))
            get_process = subprocess.CompletedProcess(args=[], returncode=0, stdout=json.dumps(original_policy))
            set_process = subprocess.CompletedProcess(args=[], returncode=1, stderr='An error.')
            ls_process2 = subprocess.CompletedProcess(args=[], returncode=1, stderr='Another error.')
            mock_run.side_effect = [ls_process, get_process, set_process, ls_process2]
            with SetEnvironmentForTest({'CLOUDSDK_CORE_PASS_CREDENTIALS_TO_GSUTIL': 'True', 'CLOUDSDK_ROOT_DIR': 'fake_dir'}):
                mock_log_handler = self.RunCommand('iam', ['ch', '-f', 'allAuthenticatedUsers:modified-role', 'gs://b/o1', 'gs://b/o2'], debug=1, return_log_handler=True)
            self.assertEqual(mock_run.call_args_list, [self._get_run_call([shim_util._get_gcloud_binary_path('fake_dir'), 'storage', 'ls', '--json', 'gs://b/o1']), self._get_run_call([shim_util._get_gcloud_binary_path('fake_dir'), 'storage', 'objects', 'get-iam-policy', 'gs://b/o1', '--format=json']), self._get_run_call([shim_util._get_gcloud_binary_path('fake_dir'), 'storage', 'objects', 'set-iam-policy', 'gs://b/o1', '-'], stdin=json.dumps(new_policy, sort_keys=True)), self._get_run_call([shim_util._get_gcloud_binary_path('fake_dir'), 'storage', 'ls', '--json', 'gs://b/o2'])])
            error_lines = '\n'.join(mock_log_handler.messages['error'])
            self.assertIn('An error.', error_lines)
            self.assertIn('Another error.', error_lines)