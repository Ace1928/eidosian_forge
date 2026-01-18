from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
from gslib.commands import hash
from gslib.exception import CommandException
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import SetEnvironmentForTest
from gslib.utils import shim_util
from six import add_move, MovedModule
from six.moves import mock
@mock.patch.object(hash.HashCommand, 'RunCommand', new=mock.Mock())
def test_shim_translates_basic_hash_command(self):
    with SetBotoConfigForTest([('GSUtil', 'use_gcloud_storage', 'True'), ('GSUtil', 'hidden_shim_mode', 'dry_run')]):
        with SetEnvironmentForTest({'CLOUDSDK_CORE_PASS_CREDENTIALS_TO_GSUTIL': 'True', 'CLOUDSDK_ROOT_DIR': 'fake_dir'}):
            mock_log_handler = self.RunCommand('hash', ['gs://b/o1', 'gs://b/o2'], return_log_handler=True)
            info_lines = '\n'.join(mock_log_handler.messages['info'])
            self.assertIn('Gcloud Storage Command: {} storage hash {} gs://b/o1 gs://b/o2'.format(shim_util._get_gcloud_binary_path('fake_dir'), hash._GCLOUD_FORMAT_STRING), info_lines)