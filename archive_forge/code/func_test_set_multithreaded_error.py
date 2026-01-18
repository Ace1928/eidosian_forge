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
def test_set_multithreaded_error(self):
    """Tests fail-fast behavior of multithreaded iam set.

    This is testing gsutil iam set with the -m and -r flags present in
    invocation.

    N.B.: Currently, (-m, -r) behaves identically to (-m, -fr) and (-fr,).
    However, (-m, -fr) and (-fr,) behavior is not as expected due to
    name_expansion.NameExpansionIterator.next raising problematic e.g. 404
    or 403 errors. More details on this issue can be found in comments in
    commands.iam.IamCommand._SetIam.

    Thus, the following command
      gsutil -m iam set -fr <object_policy> gs://bad_bucket gs://good_bucket

    will NOT set policies on objects in gs://good_bucket due to an error when
    iterating over gs://bad_bucket.
    """

    @Retry(AssertionError, tries=3, timeout_secs=1)
    def _Check1():
        stderr = self.RunGsUtil(['-m', 'iam', 'set', '-r', self.new_object_iam_path, 'gs://%s' % self.nonexistent_bucket_name, self.bucket.uri], return_stderr=True, expected_status=1)
        error_message = 'not found' if self._use_gcloud_storage else 'BucketNotFoundException'
        self.assertIn(error_message, stderr)

    @Retry(AssertionError, tries=3, timeout_secs=1)
    def _Check2():
        gsutil_object = self.CreateObject(bucket_uri=self.bucket, contents=b'foobar')
        gsutil_object2 = self.CreateObject(bucket_uri=self.bucket, contents=b'foobar')
        set_iam_string = self.RunGsUtil(['iam', 'get', gsutil_object.uri], return_stdout=True)
        set_iam_string2 = self.RunGsUtil(['iam', 'get', gsutil_object2.uri], return_stdout=True)
        self.assertEqualsPoliciesString(set_iam_string, set_iam_string2)
        self.assertEqualsPoliciesString(self.object_iam_string, set_iam_string)
    _Check1()
    _Check2()