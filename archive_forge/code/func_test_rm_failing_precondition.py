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
@SkipForS3('GCS versioning headers not supported by S3')
def test_rm_failing_precondition(self):
    """Test for '-h x-goog-if-generation-match:value rm' of an object."""
    bucket_uri = self.CreateBucket()
    object_uri = self.CreateObject(bucket_uri, contents=b'foo')
    stderr = self.RunGsUtil(['-h', 'x-goog-if-generation-match:12345', 'rm', suri(object_uri)], return_stderr=True, expected_status=1)
    if self._use_gcloud_storage:
        self.assertRegex(stderr, 'pre-conditions you specified did not hold')
    else:
        self.assertRegex(stderr, 'PreconditionException: 412')