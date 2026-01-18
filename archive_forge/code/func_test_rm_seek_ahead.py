from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
import re
import sys
from unittest import mock
from gslib.exception import NO_URLS_MATCHED_PREFIX
from gslib.exception import NO_URLS_MATCHED_TARGET
import gslib.tests.testcase as testcase
from gslib.tests.testcase.base import MAX_BUCKET_LENGTH
from gslib.tests.testcase.integration_testcase import SkipForS3
import gslib.tests.util as util
from gslib.tests.util import GenerationFromURI as urigen
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import SetEnvironmentForTest
from gslib.utils import shim_util
from gslib.utils.retry_util import Retry
def test_rm_seek_ahead(self):
    object_uri = self.CreateObject(contents=b'foo')
    with SetBotoConfigForTest([('GSUtil', 'task_estimation_threshold', '1'), ('GSUtil', 'task_estimation_force', 'True')]):
        stderr = self.RunGsUtil(['-m', 'rm', suri(object_uri)], return_stderr=True)
        self.assertIn('Estimated work for this command: objects: 1\n', stderr)