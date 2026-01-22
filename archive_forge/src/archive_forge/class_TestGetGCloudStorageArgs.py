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
class TestGetGCloudStorageArgs(testcase.GsUtilUnitTestCase):
    """Test Command.get_gcloud_storage_args method."""

    def setUp(self):
        super().setUp()
        self._fake_command = FakeCommandWithGcloudStorageMap(command_runner=mock.ANY, args=['-z', 'opt1', '-r', 'arg1', 'arg2'], headers={}, debug=1, trace_token=mock.ANY, parallel_operations=mock.ANY, bucket_storage_uri_class=mock.ANY, gsutil_api_class_map_factory=mock.MagicMock())

    def test_get_gcloud_storage_args_parses_command_and_flags(self):
        gcloud_args = self._fake_command.get_gcloud_storage_args()
        self.assertEqual(gcloud_args, ['objects', 'fake', '--zip', 'opt1', '-x', 'arg1', 'arg2'])

    def test_get_gcloud_storage_args_parses_custom_command_map(self):
        gcloud_args = self._fake_command.get_gcloud_storage_args(shim_util.GcloudStorageMap(gcloud_command=['objects', 'custom-fake'], flag_map={'-z': shim_util.GcloudStorageFlag(gcloud_flag='-a'), '-r': shim_util.GcloudStorageFlag(gcloud_flag='-b')}))
        self.assertEqual(gcloud_args, ['objects', 'custom-fake', '-a', 'opt1', '-b', 'arg1', 'arg2'])

    def test_get_gcloud_storage_args_parses_command_in_list_format(self):
        self._fake_command.gcloud_command = ['objects', 'fake']
        gcloud_args = self._fake_command.get_gcloud_storage_args()
        self.assertEqual(gcloud_args, ['objects', 'fake', '--zip', 'opt1', '-x', 'arg1', 'arg2'])

    def test_get_gcloud_storage_args_parses_subcommands(self):
        fake_with_subcommand = FakeCommandWithSubCommandWithGcloudStorageMap(command_runner=mock.ANY, args=['set', '-y', 'opt1', '-a', 'arg1', 'arg2'], headers={}, debug=mock.ANY, trace_token=mock.ANY, parallel_operations=mock.ANY, bucket_storage_uri_class=mock.ANY, gsutil_api_class_map_factory=mock.MagicMock())
        gcloud_args = fake_with_subcommand.get_gcloud_storage_args()
        self.assertEqual(gcloud_args, ['buckets', 'update', '--yyy', 'opt1', '-x', 'arg1', 'arg2'])

    def test_get_gcloud_storage_args_with_flags_to_ignore(self):
        fake_command = FakeCommandWithGcloudStorageMap(command_runner=mock.ANY, args=['positional_arg', '-f', '-r', 'opt2', '-f'], headers={}, debug=mock.ANY, trace_token=mock.ANY, parallel_operations=mock.ANY, bucket_storage_uri_class=mock.ANY, gsutil_api_class_map_factory=mock.MagicMock())
        gcloud_args = fake_command.get_gcloud_storage_args()
        self.assertEqual(gcloud_args, ['objects', 'fake', 'positional_arg', '-x', 'opt2'])

    def test_get_gcloud_storage_args_with_positional_arg_at_beginning(self):
        fake_command = FakeCommandWithGcloudStorageMap(command_runner=mock.ANY, args=['positional_arg', '-z', 'opt1', '-r', 'opt2'], headers={}, debug=mock.ANY, trace_token=mock.ANY, parallel_operations=mock.ANY, bucket_storage_uri_class=mock.ANY, gsutil_api_class_map_factory=mock.MagicMock())
        gcloud_args = fake_command.get_gcloud_storage_args()
        self.assertEqual(gcloud_args, ['objects', 'fake', 'positional_arg', '--zip', 'opt1', '-x', 'opt2'])

    def test_get_gcloud_storage_args_with_positional_arg_in_middle(self):
        fake_command = FakeCommandWithGcloudStorageMap(command_runner=mock.ANY, args=['-z', 'opt1', 'positional_arg', '-r', 'opt2'], headers={}, debug=mock.ANY, trace_token=mock.ANY, parallel_operations=mock.ANY, bucket_storage_uri_class=mock.ANY, gsutil_api_class_map_factory=mock.MagicMock())
        gcloud_args = fake_command.get_gcloud_storage_args()
        self.assertEqual(gcloud_args, ['objects', 'fake', '--zip', 'opt1', 'positional_arg', '-x', 'opt2'])

    def test_get_gcloud_storage_args_with_repeat_flag_list(self):
        fake_command = FakeCommandWithGcloudStorageMap(command_runner=mock.ANY, args=['-l', 'flag_value1', '-l', 'flag_value2', 'positional_arg'], headers={}, debug=mock.ANY, trace_token=mock.ANY, parallel_operations=mock.ANY, bucket_storage_uri_class=mock.ANY, gsutil_api_class_map_factory=mock.MagicMock())
        gcloud_args = fake_command.get_gcloud_storage_args()
        self.assertEqual(gcloud_args, ['objects', 'fake', 'positional_arg', '--ludicrous-list=flag_value1,flag_value2'])

    def test_get_gcloud_storage_args_with_repeat_flag_dict(self):
        fake_command = FakeCommandWithGcloudStorageMap(command_runner=mock.ANY, args=['-d', 'flag_key1:flag_value1', '-d', 'flag_key2:flag_value2', 'positional_arg'], headers={}, debug=mock.ANY, trace_token=mock.ANY, parallel_operations=mock.ANY, bucket_storage_uri_class=mock.ANY, gsutil_api_class_map_factory=mock.MagicMock())
        gcloud_args = fake_command.get_gcloud_storage_args()
        self.assertEqual(gcloud_args, ['objects', 'fake', 'positional_arg', '--delightful-dict=flag_key1=flag_value1,flag_key2=flag_value2'])

    def test_get_gcloud_storage_args_with_value_translated_to_flag(self):
        fake_command = FakeCommandWithGcloudStorageMap(command_runner=mock.ANY, args=['-e', 'on', 'positional_arg'], headers={}, debug=mock.ANY, trace_token=mock.ANY, parallel_operations=mock.ANY, bucket_storage_uri_class=mock.ANY, gsutil_api_class_map_factory=mock.MagicMock())
        gcloud_args = fake_command.get_gcloud_storage_args()
        self.assertEqual(gcloud_args, ['objects', 'fake', '--e-on', 'positional_arg'])
        fake_command = FakeCommandWithGcloudStorageMap(command_runner=mock.ANY, args=['positional_arg', '-e', 'off'], headers={}, debug=mock.ANY, trace_token=mock.ANY, parallel_operations=mock.ANY, bucket_storage_uri_class=mock.ANY, gsutil_api_class_map_factory=mock.MagicMock())
        gcloud_args = fake_command.get_gcloud_storage_args()
        self.assertEqual(gcloud_args, ['objects', 'fake', 'positional_arg', '--e-off'])

    def test_raises_error_for_invalid_value_translated_to_flag(self):
        fake_command = FakeCommandWithGcloudStorageMap(command_runner=mock.ANY, args=['-e', 'incorrect', 'positional_arg'], headers={}, debug=mock.ANY, trace_token=mock.ANY, parallel_operations=mock.ANY, bucket_storage_uri_class=mock.ANY, gsutil_api_class_map_factory=mock.MagicMock())
        with self.assertRaisesRegex(ValueError, 'Flag value not in translation map for "-e": incorrect'):
            gcloud_args = fake_command.get_gcloud_storage_args()

    def test_raises_error_if_gcloud_storage_map_is_missing(self):
        self._fake_command.gcloud_storage_map = None
        with self.assertRaisesRegex(exception.GcloudStorageTranslationError, 'Command "fake_shim" cannot be translated to gcloud storage because the translation mapping is missing'):
            self._fake_command.get_gcloud_storage_args()

    def test_raises_error_if_gcloud_command_is_of_incorrect_type(self):
        self._fake_command.gcloud_storage_map = shim_util.GcloudStorageMap(gcloud_command='some fake command as a string', flag_map={})
        with self.assertRaisesRegex(ValueError, 'Incorrect mapping found for "fake_shim" command'):
            self._fake_command.get_gcloud_storage_args()

    def test_raises_error_if_command_option_mapping_is_missing(self):
        self._fake_command.gcloud_storage_map = shim_util.GcloudStorageMap(gcloud_command=['fake'], flag_map={'-z': shim_util.GcloudStorageFlag('-a')})
        with self.assertRaisesRegex(exception.GcloudStorageTranslationError, 'Command option "-r" cannot be translated to gcloud storage'):
            self._fake_command.get_gcloud_storage_args()

    def test_raises_error_if_sub_command_mapping_is_missing(self):
        fake_with_subcommand = FakeCommandWithSubCommandWithGcloudStorageMap(command_runner=mock.ANY, args=['set', '-y', 'opt1', '-a', 'arg1', 'arg2'], headers={}, debug=mock.ANY, trace_token=mock.ANY, parallel_operations=mock.ANY, bucket_storage_uri_class=mock.ANY, gsutil_api_class_map_factory=mock.MagicMock())
        fake_with_subcommand.gcloud_storage_map = shim_util.GcloudStorageMap(gcloud_command={}, flag_map={})
        with self.assertRaisesRegex(exception.GcloudStorageTranslationError, 'Command "fake_with_sub" cannot be translated to gcloud storage because the translation mapping is missing.'):
            fake_with_subcommand.get_gcloud_storage_args()

    def test_raises_error_if_flags_mapping_at_top_level_for_subcommand(self):
        fake_with_subcommand = FakeCommandWithSubCommandWithGcloudStorageMap(command_runner=mock.ANY, args=['set', '-y', 'opt1', '-a', 'arg1', 'arg2'], headers={}, debug=mock.ANY, trace_token=mock.ANY, parallel_operations=mock.ANY, bucket_storage_uri_class=mock.ANY, gsutil_api_class_map_factory=mock.MagicMock())
        fake_with_subcommand.gcloud_storage_map.flag_map = {'a': 'b'}
        with self.assertRaisesRegex(ValueError, 'Flags mapping should not be present at the top-level command if a sub-command is used. Command: fake_with_sub'):
            fake_with_subcommand.get_gcloud_storage_args()