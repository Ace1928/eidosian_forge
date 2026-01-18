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
def test_some_missing(self):
    """Test that 'rm -a' fails when some but not all uris don't exist."""
    bucket_uri = self.CreateVersionedBucket()
    key_uri = self.StorageUriCloneReplaceName(bucket_uri, 'foo')
    self.StorageUriSetContentsFromString(key_uri, 'bar')
    if self.multiregional_buckets:
        self.AssertNObjectsInBucket(bucket_uri, 1, versioned=True)
    stderr = self.RunGsUtil(['rm', '-a', suri(key_uri), '%s' % suri(bucket_uri, 'missing')], return_stderr=True, expected_status=1)
    self.assertEqual(stderr.count('Removing %s://' % self.default_provider), 1)
    if self._use_gcloud_storage:
        self.assertIn('The following URLs matched no objects or files:\n-%s' % suri(bucket_uri, 'missing'), stderr)
    else:
        self.assertIn(NO_URLS_MATCHED_TARGET % suri(bucket_uri, 'missing'), stderr)