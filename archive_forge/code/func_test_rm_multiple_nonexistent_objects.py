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
def test_rm_multiple_nonexistent_objects(self):
    bucket_uri = self.CreateBucket()
    nonexistent_object1 = suri(bucket_uri, 'nonexistent1')
    nonexistent_object2 = suri(bucket_uri, 'nonexistent2')
    stderr = self.RunGsUtil(['rm', '-rf', nonexistent_object1, nonexistent_object2], return_stderr=True, expected_status=1)
    if self._use_gcloud_storage:
        self.assertIn('The following URLs matched no objects or files:\n-{}\n-{}'.format(nonexistent_object1, nonexistent_object2), stderr)
    else:
        self.assertIn('2 files/objects could not be removed.', stderr)