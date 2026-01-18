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
def test_set_valid_iam_multiple_objects(self):
    """Tests setting a valid IAM on multiple objects."""
    old_gsutil_object, old_gsutil_object2, gsutil_object, gsutil_object2 = self._create_multiple_objects()
    self.RunGsUtil(['iam', 'set', '-r', self.new_object_iam_path, self.versioned_bucket.uri])
    set_iam_string = self.RunGsUtil(['iam', 'get', gsutil_object.uri], return_stdout=True)
    set_iam_string2 = self.RunGsUtil(['iam', 'get', gsutil_object2.uri], return_stdout=True)
    self.assertEqualsPoliciesString(set_iam_string, set_iam_string2)
    self.assertIn(self.public_object_read_binding[0], json.loads(set_iam_string)['bindings'])
    iam_string_old = self.RunGsUtil(['iam', 'get', old_gsutil_object.version_specific_uri], return_stdout=True)
    iam_string_old2 = self.RunGsUtil(['iam', 'get', old_gsutil_object2.version_specific_uri], return_stdout=True)
    self.assertEqualsPoliciesString(iam_string_old, iam_string_old2)
    self.assertEqualsPoliciesString(self.object_iam_string, iam_string_old)