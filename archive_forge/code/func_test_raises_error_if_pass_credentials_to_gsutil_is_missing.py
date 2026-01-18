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
def test_raises_error_if_pass_credentials_to_gsutil_is_missing(self):
    error_regex = 'CommandException: Requested to use "gcloud storage" but gsutil is not using the same credentials as gcloud. You can make gsutil use the same credentials by running:\\n{} config set pass_credentials_to_gsutil True'.format(re.escape(shim_util._get_gcloud_binary_path('fake_dir')))
    with util.SetBotoConfigForTest([('GSUtil', 'use_gcloud_storage', 'True'), ('GSUtil', 'hidden_shim_mode', 'no_fallback')]):
        with util.SetEnvironmentForTest({'CLOUDSDK_ROOT_DIR': 'fake_dir', 'CLOUDSDK_CORE_PASS_CREDENTIALS_TO_GSUTIL': None}):
            with self.assertRaisesRegex(exception.CommandException, error_regex):
                self._fake_command.translate_to_gcloud_storage_if_requested()