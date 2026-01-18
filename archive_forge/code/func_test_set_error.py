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
def test_set_error(self):
    """Tests fail-fast behavior of iam set.

    We initialize two buckets (bucket, bucket2) and attempt to set both along
    with a third, non-existent bucket in between, self.nonexistent_bucket_name.

    We want to ensure
      1.) Bucket "bucket" IAM policy has been set appropriately,
      2.) Bucket self.nonexistent_bucket_name has caused an error, and
      3.) gsutil has exited and "bucket2"'s IAM policy is unaltered.
    """
    bucket = self.CreateBucket()
    bucket2 = self.CreateBucket()
    stderr = self.RunGsUtil(['iam', 'set', '-e', '', self.new_bucket_iam_path, bucket.uri, 'gs://%s' % self.nonexistent_bucket_name, bucket2.uri], return_stderr=True, expected_status=1)
    error_message = 'not found' if self._use_gcloud_storage else 'BucketNotFoundException'
    self.assertIn(error_message, stderr)
    set_iam_string = self.RunGsUtil(['iam', 'get', bucket.uri], return_stdout=True)
    set_iam_string2 = self.RunGsUtil(['iam', 'get', bucket2.uri], return_stdout=True)
    self.assertIn(self.public_bucket_read_binding[0], json.loads(set_iam_string)['bindings'])
    self.assertEqualsPoliciesString(self.bucket_iam_string, set_iam_string2)