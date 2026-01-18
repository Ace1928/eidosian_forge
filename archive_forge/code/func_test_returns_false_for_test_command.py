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
def test_returns_false_for_test_command(self):
    command = test.TestCommand(command_runner=mock.ANY, args=[], headers={}, debug=0, trace_token=None, parallel_operations=True, bucket_storage_uri_class=mock.ANY, gsutil_api_class_map_factory=mock.MagicMock())
    with util.SetBotoConfigForTest([('GSUtil', 'use_gcloud_storage', 'True'), ('GSUtil', 'hidden_shim_mode', 'no_fallback')]):
        with mock.patch.object(command, 'get_gcloud_storage_args', autospec=True) as mock_get_gcloud_storage_args:
            self.assertFalse(command.translate_to_gcloud_storage_if_requested())
            self.assertFalse(mock_get_gcloud_storage_args.called)