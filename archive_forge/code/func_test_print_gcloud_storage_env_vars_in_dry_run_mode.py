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
def test_print_gcloud_storage_env_vars_in_dry_run_mode(self):
    """Should log the command and env vars to logger.info"""
    with mock.patch.object(self._fake_command, 'logger', autospec=True) as mock_logger:
        self._fake_command._print_gcloud_storage_command_info(['fake', 'gcloud', 'command'], {'fake_env_var': 'val'}, dry_run=True)
        expected_calls = [mock.call('Gcloud Storage Command: fake gcloud command'), mock.call('Environment variables for Gcloud Storage:'), mock.call('%s=%s', 'fake_env_var', 'val')]
        self.assertEqual(mock_logger.info.mock_calls, expected_calls)