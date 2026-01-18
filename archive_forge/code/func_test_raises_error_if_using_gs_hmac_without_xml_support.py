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
@mock.patch.object(boto_util, 'UsingGsHmac', return_value=True)
def test_raises_error_if_using_gs_hmac_without_xml_support(self, _):
    with util.SetBotoConfigForTest([('GSUtil', 'use_gcloud_storage', 'True'), ('GSUtil', 'hidden_shim_mode', 'no_fallback')]):
        with util.SetEnvironmentForTest({'CLOUDSDK_ROOT_DIR': 'fake_dir', 'CLOUDSDK_CORE_PASS_CREDENTIALS_TO_GSUTIL': 'True'}):
            self._fake_command.command_spec = command.Command.CreateCommandSpec('fake_shim', gs_api_support=[ApiSelector.JSON])
            with self.assertRaisesRegex(exception.CommandException, 'CommandException: Requested to use "gcloud storage" with Cloud Storage XML API HMAC credentials but the "fake_shim" command can only be used with the Cloud Storage JSON API.'):
                self._fake_command.translate_to_gcloud_storage_if_requested()