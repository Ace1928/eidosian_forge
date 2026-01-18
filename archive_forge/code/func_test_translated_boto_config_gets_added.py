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
@mock.patch.object(shim_util, 'COMMANDS_SUPPORTING_ALL_HEADERS', new={'fake_shim'})
def test_translated_boto_config_gets_added(self):
    """Should add translated env vars as well flags."""
    with _mock_boto_config({'GSUtil': {'use_gcloud_storage': 'True', 'hidden_shim_mode': 'no_fallback', 'content_language': 'foo', 'default_project_id': 'fake_project'}}):
        with util.SetEnvironmentForTest({'CLOUDSDK_CORE_PASS_CREDENTIALS_TO_GSUTIL': 'True', 'CLOUDSDK_ROOT_DIR': 'fake_dir'}):
            self.assertTrue(self._fake_command.translate_to_gcloud_storage_if_requested())
            expected_gcloud_path = shim_util._get_gcloud_binary_path('fake_dir')
            self.assertEqual(self._fake_command._translated_gcloud_storage_command, [expected_gcloud_path, 'objects', 'fake', '--zip', 'opt1', '-x', 'arg1', 'arg2', '--content-language=foo'])
            self.assertEqual(self._fake_command._translated_env_variables, {'CLOUDSDK_CORE_PROJECT': 'fake_project', 'CLOUDSDK_METRICS_ENVIRONMENT': 'gsutil_shim', 'CLOUDSDK_STORAGE_RUN_BY_GSUTIL_SHIM': 'True'})