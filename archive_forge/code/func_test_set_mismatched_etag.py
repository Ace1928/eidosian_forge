from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from collections import defaultdict
import json
import os
import subprocess
from gslib.commands import iam
from gslib.exception import CommandException
from gslib.project_id import PopulateProjectId
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.testcase.integration_testcase import SkipForXML
from gslib.tests.util import GenerationFromURI as urigen
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import SetEnvironmentForTest
from gslib.tests.util import unittest
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
from gslib.utils import shim_util
from gslib.utils.constants import UTF8
from gslib.utils.iam_helper import BindingsMessageToUpdateDict
from gslib.utils.iam_helper import BindingsDictToUpdateDict
from gslib.utils.iam_helper import BindingStringToTuple as bstt
from gslib.utils.iam_helper import DiffBindings
from gslib.utils.iam_helper import IsEqualBindings
from gslib.utils.iam_helper import PatchBindings
from gslib.utils.retry_util import Retry
from six import add_move, MovedModule
from six.moves import mock
def test_set_mismatched_etag(self):
    """Tests setting mismatched etag raises an error."""
    get_iam_string = self.RunGsUtil(['iam', 'get', self.bucket.uri], return_stdout=True)
    self.RunGsUtil(['iam', 'set', '-e', json.loads(get_iam_string)['etag'], self.new_bucket_iam_path, self.bucket.uri])
    stderr = self.RunGsUtil(['iam', 'set', '-e', json.loads(get_iam_string)['etag'], self.new_bucket_iam_path, self.bucket.uri], return_stderr=True, expected_status=1)
    error_message = 'pre-conditions you specified did not hold' if self._use_gcloud_storage else 'PreconditionException'
    self.assertIn(error_message, stderr)