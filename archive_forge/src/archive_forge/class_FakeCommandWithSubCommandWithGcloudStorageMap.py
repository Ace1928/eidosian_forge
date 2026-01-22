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
class FakeCommandWithSubCommandWithGcloudStorageMap(command.Command):
    """Implementation of a fake gsutil command."""
    command_spec = command.Command.CreateCommandSpec('fake_with_sub', min_args=1, max_args=constants.NO_MAX, supported_sub_args='ay:', file_url_ok=True, argparse_arguments={'set': [command_argument.CommandArgument.MakeZeroOrMoreCloudBucketURLsArgument()], 'get': [command_argument.CommandArgument.MakeNCloudBucketURLsArgument(1)]})
    gcloud_storage_map = shim_util.GcloudStorageMap(gcloud_command={'set': shim_util.GcloudStorageMap(gcloud_command=['buckets', 'update'], flag_map={'-a': shim_util.GcloudStorageFlag(gcloud_flag='-x'), '-y': shim_util.GcloudStorageFlag(gcloud_flag='--yyy')}), 'get': shim_util.GcloudStorageMap(gcloud_command=['buckets', 'describe'], flag_map={})}, flag_map={})
    help_spec = command.Command.HelpSpec(help_name='fake_with_sub', help_name_aliases=[], help_type='command_help', help_one_line_summary='Fake one line summary for the command.', help_text='Help text for fake command with sub commands.', subcommand_help_text={})