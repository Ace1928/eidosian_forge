from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
import re
import six
from gslib.commands import defacl
from gslib.cs_api_map import ApiSelector
import gslib.tests.testcase as case
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import SetEnvironmentForTest
from gslib.tests.util import unittest
from gslib.utils.constants import UTF8
from gslib.utils import shim_util
from six import add_move, MovedModule
from six.moves import mock
class TestDefaclShim(case.GsUtilUnitTestCase):

    @mock.patch.object(defacl.DefAclCommand, 'RunCommand', new=mock.Mock())
    def test_shim_translates_defacl_get(self):
        with SetBotoConfigForTest([('GSUtil', 'use_gcloud_storage', 'True'), ('GSUtil', 'hidden_shim_mode', 'dry_run')]):
            with SetEnvironmentForTest({'CLOUDSDK_CORE_PASS_CREDENTIALS_TO_GSUTIL': 'True', 'CLOUDSDK_ROOT_DIR': 'fake_dir'}):
                mock_log_handler = self.RunCommand('defacl', ['get', 'gs://bucket'], return_log_handler=True)
                info_lines = '\n'.join(mock_log_handler.messages['info'])
                self.assertIn('Gcloud Storage Command: {} storage buckets describe --format=multi(defaultObjectAcl:format=json) --raw gs://bucket'.format(shim_util._get_gcloud_binary_path('fake_dir')), info_lines)

    @mock.patch.object(defacl.DefAclCommand, 'RunCommand', new=mock.Mock())
    def test_shim_translates_set_defacl_file(self):
        acl_string = 'acl_string'
        inpath = self.CreateTempFile(contents=acl_string.encode(UTF8))
        with SetBotoConfigForTest([('GSUtil', 'use_gcloud_storage', 'True'), ('GSUtil', 'hidden_shim_mode', 'dry_run')]):
            with SetEnvironmentForTest({'CLOUDSDK_CORE_PASS_CREDENTIALS_TO_GSUTIL': 'True', 'CLOUDSDK_ROOT_DIR': 'fake_dir'}):
                mock_log_handler = self.RunCommand('defacl', ['set', inpath, 'gs://b1', 'gs://b2'], return_log_handler=True)
                info_lines = '\n'.join(mock_log_handler.messages['info'])
                self.assertIn('Gcloud Storage Command: {} storage buckets update --default-object-acl-file={} gs://b1 gs://b2'.format(shim_util._get_gcloud_binary_path('fake_dir'), inpath), info_lines)

    @mock.patch.object(defacl.DefAclCommand, 'RunCommand', new=mock.Mock())
    def test_shim_translates_set_predefined_defacl(self):
        with SetBotoConfigForTest([('GSUtil', 'use_gcloud_storage', 'True'), ('GSUtil', 'hidden_shim_mode', 'dry_run')]):
            with SetEnvironmentForTest({'CLOUDSDK_CORE_PASS_CREDENTIALS_TO_GSUTIL': 'True', 'CLOUDSDK_ROOT_DIR': 'fake_dir'}):
                mock_log_handler = self.RunCommand('defacl', ['set', 'bucket-owner-read', 'gs://b1', 'gs://b2'], return_log_handler=True)
                info_lines = '\n'.join(mock_log_handler.messages['info'])
                self.assertIn('Gcloud Storage Command: {} storage buckets update --predefined-default-object-acl={} gs://b1 gs://b2'.format(shim_util._get_gcloud_binary_path('fake_dir'), 'bucketOwnerRead'), info_lines)

    @mock.patch.object(defacl.DefAclCommand, 'RunCommand', new=mock.Mock())
    def test_shim_translates_xml_predefined_defacl_for_set(self):
        with SetBotoConfigForTest([('GSUtil', 'use_gcloud_storage', 'True'), ('GSUtil', 'hidden_shim_mode', 'dry_run')]):
            with SetEnvironmentForTest({'CLOUDSDK_CORE_PASS_CREDENTIALS_TO_GSUTIL': 'True', 'CLOUDSDK_ROOT_DIR': 'fake_dir'}):
                mock_log_handler = self.RunCommand('defacl', ['set', 'authenticated-read', 'gs://b1', 'gs://b2'], return_log_handler=True)
                info_lines = '\n'.join(mock_log_handler.messages['info'])
                self.assertIn('Gcloud Storage Command: {} storage buckets update --predefined-default-object-acl={} gs://b1 gs://b2'.format(shim_util._get_gcloud_binary_path('fake_dir'), 'authenticatedRead'), info_lines)

    @mock.patch.object(defacl.DefAclCommand, 'RunCommand', new=mock.Mock())
    def test_shim_changes_defacls_for_user(self):
        inpath = self.CreateTempFile()
        with SetBotoConfigForTest([('GSUtil', 'use_gcloud_storage', 'True'), ('GSUtil', 'hidden_shim_mode', 'dry_run')]):
            with SetEnvironmentForTest({'CLOUDSDK_CORE_PASS_CREDENTIALS_TO_GSUTIL': 'True', 'CLOUDSDK_ROOT_DIR': 'fake_dir'}):
                mock_log_handler = self.RunCommand('defacl', ['ch', '-f', '-u', 'user@example.com:R', 'gs://bucket1'], return_log_handler=True)
                info_lines = '\n'.join(mock_log_handler.messages['info'])
                self.assertIn('Gcloud Storage Command: {} storage buckets update --continue-on-error --add-default-object-acl-grant entity=user-user@example.com,role=READER gs://bucket1'.format(shim_util._get_gcloud_binary_path('fake_dir'), inpath), info_lines)