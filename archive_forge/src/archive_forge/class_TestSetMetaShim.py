from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
import re
from unittest import mock
import six
from gslib.commands import setmeta
from gslib.cs_api_map import ApiSelector
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import SetEnvironmentForTest
from gslib.tests.util import unittest
from gslib.utils.retry_util import Retry
from gslib.utils import shim_util
class TestSetMetaShim(testcase.GsUtilUnitTestCase):

    @mock.patch.object(setmeta.SetMetaCommand, 'RunCommand', new=mock.Mock())
    def test_shim_translates_setmeta_set_and_clear_flags(self):
        with SetBotoConfigForTest([('GSUtil', 'use_gcloud_storage', 'True'), ('GSUtil', 'hidden_shim_mode', 'dry_run')]):
            with SetEnvironmentForTest({'CLOUDSDK_CORE_PASS_CREDENTIALS_TO_GSUTIL': 'True', 'CLOUDSDK_ROOT_DIR': 'fake_dir'}):
                mock_log_handler = self.RunCommand('setmeta', ['-r', '-h', 'Cache-Control:', '-h', 'Content-Type:fake-content-type', 'gs://bucket/object'], return_log_handler=True)
                info_lines = '\n'.join(mock_log_handler.messages['info'])
                self.assertIn('Gcloud Storage Command: {} storage objects update --recursive --clear-cache-control --content-type=fake-content-type gs://bucket/object'.format(shim_util._get_gcloud_binary_path('fake_dir')), info_lines)