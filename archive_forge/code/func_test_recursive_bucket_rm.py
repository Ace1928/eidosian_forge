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
@SkipForS3('Listing/removing S3 DeleteMarkers is not supported')
def test_recursive_bucket_rm(self):
    """Test for 'rm -r' of a bucket."""
    bucket_uri = self.CreateBucket()
    object_uri = self.CreateObject(bucket_uri, contents=b'foo')
    if self.multiregional_buckets:
        self.AssertNObjectsInBucket(bucket_uri, 1)
    self._RunRemoveCommandAndCheck(['rm', '-r', suri(bucket_uri)], objects_to_remove=['%s#%s' % (suri(object_uri), urigen(object_uri))], buckets_to_remove=[suri(bucket_uri)])

    @Retry(AssertionError, tries=3, timeout_secs=1)
    def _Check1():
        stderr = self.RunGsUtil(['ls', '-Lb', suri(bucket_uri)], return_stderr=True, expected_status=1, force_gsutil=True)
        self.assertIn('bucket does not exist', stderr)
    _Check1()
    bucket_uri = self.CreateVersionedBucket()
    self.CreateObject(bucket_uri, 'obj', contents=b'z')
    self.CreateObject(bucket_uri, 'obj', contents=b'z')
    final_uri = self.CreateObject(bucket_uri, 'obj', contents=b'z')
    if self.multiregional_buckets:
        self.AssertNObjectsInBucket(bucket_uri, 3, versioned=True)
    self._RunRemoveCommandAndCheck(['rm', suri(bucket_uri, '**')], objects_to_remove=['%s' % final_uri])
    stderr = self.RunGsUtil(['rb', suri(bucket_uri)], return_stderr=True, expected_status=1, force_gsutil=True)
    self.assertIn('not empty', stderr)

    @Retry(AssertionError, tries=3, timeout_secs=1)
    def _Check2():
        self.RunGsUtil(['rm', '-r', suri(bucket_uri)])
        stderr = self.RunGsUtil(['ls', '-Lb', suri(bucket_uri)], return_stderr=True, expected_status=1, force_gsutil=True)
        self.assertIn('bucket does not exist', stderr)
    _Check2()