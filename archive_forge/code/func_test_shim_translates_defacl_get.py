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
@mock.patch.object(defacl.DefAclCommand, 'RunCommand', new=mock.Mock())
def test_shim_translates_defacl_get(self):
    with SetBotoConfigForTest([('GSUtil', 'use_gcloud_storage', 'True'), ('GSUtil', 'hidden_shim_mode', 'dry_run')]):
        with SetEnvironmentForTest({'CLOUDSDK_CORE_PASS_CREDENTIALS_TO_GSUTIL': 'True', 'CLOUDSDK_ROOT_DIR': 'fake_dir'}):
            mock_log_handler = self.RunCommand('defacl', ['get', 'gs://bucket'], return_log_handler=True)
            info_lines = '\n'.join(mock_log_handler.messages['info'])
            self.assertIn('Gcloud Storage Command: {} storage buckets describe --format=multi(defaultObjectAcl:format=json) --raw gs://bucket'.format(shim_util._get_gcloud_binary_path('fake_dir')), info_lines)