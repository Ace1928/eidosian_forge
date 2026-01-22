from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import collections
from contextlib import contextmanager
import os
import re
import subprocess
from unittest import mock
from boto import config
from gslib import command
from gslib import command_argument
from gslib import exception
from gslib.commands import rsync
from gslib.commands import version
from gslib.commands import test
from gslib.cs_api_map import ApiSelector
from gslib.tests import testcase
from gslib.utils import boto_util
from gslib.utils import constants
from gslib.utils import shim_util
from gslib.utils import system_util
from gslib.tests import util
class TestHeaderTranslation(testcase.GsUtilUnitTestCase):
    """Test gsutil header  translation."""

    def setUp(self):
        super().setUp()
        self._fake_command = FakeCommandWithGcloudStorageMap(command_runner=mock.ANY, args=['-z', 'opt1', '-r', 'arg1', 'arg2'], headers={}, debug=0, trace_token=None, parallel_operations=True, bucket_storage_uri_class=mock.ANY, gsutil_api_class_map_factory=mock.MagicMock())

    @mock.patch.object(shim_util, 'COMMANDS_SUPPORTING_ALL_HEADERS', new={'fake_shim'})
    def test_translated_headers_get_added_to_final_command(self):
        with _mock_boto_config({'GSUtil': {'use_gcloud_storage': 'always', 'hidden_shim_mode': 'no_fallback'}}):
            with util.SetEnvironmentForTest({'CLOUDSDK_CORE_PASS_CREDENTIALS_TO_GSUTIL': 'True', 'CLOUDSDK_ROOT_DIR': 'fake_dir'}):
                fake_command = FakeCommandWithGcloudStorageMap(command_runner=mock.ANY, args=['arg1', 'arg2'], headers={'Content-Type': 'fake_val'}, debug=1, trace_token=None, parallel_operations=mock.ANY, bucket_storage_uri_class=mock.ANY, gsutil_api_class_map_factory=mock.MagicMock())
                self.assertTrue(fake_command.translate_to_gcloud_storage_if_requested())
                self.assertEqual(fake_command._translated_gcloud_storage_command, [shim_util._get_gcloud_binary_path('fake_dir'), 'objects', 'fake', 'arg1', 'arg2', '--content-type=fake_val'])

    @mock.patch.object(shim_util, 'COMMANDS_SUPPORTING_ALL_HEADERS', new={'fake_shim'})
    def test_translate_headers_returns_correct_flags_for_data_transfer_command(self):
        self._fake_command.headers = {'Cache-Control': 'fake_Cache_Control', 'Content-Disposition': 'fake_Content-Disposition', 'Content-Encoding': 'fake_Content-Encoding', 'Content-Language': 'fake_Content-Language', 'Content-Type': 'fake_Content-Type', 'Content-MD5': 'fake_Content-MD5', 'custom-time': 'fake_time', 'x-goog-if-generation-match': 'fake_gen_match', 'x-goog-if-metageneration-match': 'fake_metagen_match', 'x-goog-meta-cAsE': 'sEnSeTiVe', 'x-goog-meta-gfoo': 'fake_goog_meta', 'x-amz-meta-afoo': 'fake_amz_meta', 'x-amz-afoo': 'fake_amz_custom_header'}
        flags = self._fake_command._translate_headers()
        self.assertCountEqual(flags, ['--cache-control=fake_Cache_Control', '--content-disposition=fake_Content-Disposition', '--content-encoding=fake_Content-Encoding', '--content-language=fake_Content-Language', '--content-type=fake_Content-Type', '--content-md5=fake_Content-MD5', '--custom-time=fake_time', '--if-generation-match=fake_gen_match', '--if-metageneration-match=fake_metagen_match', '--update-custom-metadata=cAsE=sEnSeTiVe', '--update-custom-metadata=gfoo=fake_goog_meta', '--update-custom-metadata=afoo=fake_amz_meta', '--additional-headers=x-amz-afoo=fake_amz_custom_header'])

    @mock.patch.object(shim_util, 'COMMANDS_SUPPORTING_ALL_HEADERS', new={'fake_shim'})
    def test_translate_custom_headers_returns_correct_flags(self):
        flags = self._fake_command._translate_headers({'Cache-Control': 'fake_Cache_Control'})
        self.assertCountEqual(flags, ['--cache-control=fake_Cache_Control'])

    @mock.patch.object(shim_util, 'COMMANDS_SUPPORTING_ALL_HEADERS', new={'fake_shim'})
    def test_translate_custom_headers_handles_multiple_additional_headers(self):
        flags = self._fake_command._translate_headers(collections.OrderedDict([('header1', 'value1'), ('header2', 'value2')]))
        self.assertCountEqual(flags, ['--additional-headers=header1=value1,header2=value2'])

    @mock.patch.object(shim_util, 'COMMANDS_SUPPORTING_ALL_HEADERS', new={'fake_shim'})
    def test_translate_clear_headers_returns_correct_flags(self):
        flags = self._fake_command._translate_headers({'Cache-Control': 'fake_Cache_Control'}, unset=True)
        self.assertCountEqual(flags, ['--clear-cache-control'])

    @mock.patch.object(shim_util, 'COMMANDS_SUPPORTING_ALL_HEADERS', new={'fake_shim'})
    def test_translate_headers_for_data_transfer_command_with_additional_header(self):
        """Should log a warning."""
        self._fake_command.headers = {'additional': 'header'}
        with mock.patch.object(self._fake_command.logger, 'warn', autospec=True) as mock_warning:
            self.assertEqual(self._fake_command._translate_headers(), ['--additional-headers=additional=header'])
            mock_warning.assert_called_once_with('Header additional:header cannot be translated to a gcloud storage equivalent flag. It is being treated as an arbitrary request header.')

    @mock.patch.object(shim_util, 'PRECONDITONS_ONLY_SUPPORTED_COMMANDS', new={'fake_shim'})
    def test_translate_valid_headers_for_precondition_supported_command(self):
        self._fake_command.headers = {'x-goog-if-generation-match': 'fake_gen_match', 'x-goog-if-metageneration-match': 'fake_metagen_match', 'x-goog-meta-foo': 'fake_goog_meta'}
        flags = self._fake_command._translate_headers()
        self.assertCountEqual(flags, ['--if-generation-match=fake_gen_match', '--if-metageneration-match=fake_metagen_match'])

    @mock.patch.object(shim_util, 'PRECONDITONS_ONLY_SUPPORTED_COMMANDS', new={'fake_shim'})
    def test_translate_short_headers_for_precondition_supported_command(self):
        self._fake_command.headers = {'x-goog-generation-match': 'fake_gen_match', 'x-goog-metageneration-match': 'fake_metagen_match', 'x-goog-meta-foo': 'fake_goog_meta'}
        flags = self._fake_command._translate_headers()
        self.assertCountEqual(flags, ['--if-generation-match=fake_gen_match', '--if-metageneration-match=fake_metagen_match'])

    @mock.patch.object(shim_util, 'PRECONDITONS_ONLY_SUPPORTED_COMMANDS', new={'fake_shim'})
    def test_translate_headers_for_precondition_supported_command_with_additional_header(self):
        """Should be ignored and not raise any error."""
        self._fake_command.headers = {'additional': 'header'}
        with mock.patch.object(self._fake_command.logger, 'warn', autospec=True) as mock_warning:
            self.assertEqual(self._fake_command._translate_headers(), ['--additional-headers=additional=header'])
            mock_warning.assert_called_once_with('Header additional:header cannot be translated to a gcloud storage equivalent flag. It is being treated as an arbitrary request header.')

    def test_translate_headers_only_uses_additional_headers_for_commands_not_in_allowlist(self):
        self._fake_command.headers = {'Cache-Control': 'fake_Cache_Control', 'x-goog-if-generation-match': 'fake_gen_match', 'x-goog-meta-foo': 'fake_goog_meta', 'additional': 'header'}
        self.assertEqual(self._fake_command._translate_headers(), ['--additional-headers=additional=header'])

    @mock.patch.object(shim_util, 'PRECONDITONS_ONLY_SUPPORTED_COMMANDS', new={'fake_shim'})
    def test_translate_headers_ignores_x_goog_api_version_header(self):
        self._fake_command.headers = {'x-goog-if-generation-match': 'fake_gen_match', 'x-goog-api-version': '2'}
        self.assertEqual(self._fake_command._translate_headers(), ['--if-generation-match=fake_gen_match'])