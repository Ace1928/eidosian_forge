from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import json
import os
import posixpath
from unittest import mock
from xml.dom.minidom import parseString
from gslib.cs_api_map import ApiSelector
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import SetEnvironmentForTest
from gslib.tests.util import unittest
from gslib.utils.retry_util import Retry
from gslib.utils.translation_helper import LifecycleTranslation
from gslib.utils import shim_util
def test_set_lifecycle_wildcard(self):
    """Tests setting lifecycle with a wildcarded bucket URI."""
    if self.test_api == ApiSelector.XML:
        return unittest.skip('XML wildcard behavior can cause test to flake if a bucket in the same project is deleted during execution.')
    random_prefix = self.MakeRandomTestString()
    bucket1_name = self.MakeTempName('bucket', prefix=random_prefix)
    bucket2_name = self.MakeTempName('bucket', prefix=random_prefix)
    bucket1_uri = self.CreateBucket(bucket_name=bucket1_name)
    bucket2_uri = self.CreateBucket(bucket_name=bucket2_name)
    common_prefix = posixpath.commonprefix([suri(bucket1_uri), suri(bucket2_uri)])
    self.assertTrue(common_prefix.startswith('gs://%sgsutil-test-test-set-lifecycle-wildcard-' % random_prefix))
    wildcard = '%s*' % common_prefix
    fpath = self.CreateTempFile(contents=self.lifecycle_doc.encode('ascii'))
    actual_lines = set()

    @Retry(AssertionError, tries=3, timeout_secs=1)
    def _Check1():
        stderr = self.RunGsUtil(['lifecycle', 'set', fpath, wildcard], return_stderr=True)
        actual_lines.update(stderr.splitlines())
        if self._use_gcloud_storage:
            self.assertIn('Updating %s/...' % suri(bucket1_uri), stderr)
            self.assertIn('Updating %s/...' % suri(bucket2_uri), stderr)
            status_message = 'Updating'
        else:
            expected_lines = set(['Setting lifecycle configuration on %s/...' % suri(bucket1_uri), 'Setting lifecycle configuration on %s/...' % suri(bucket2_uri)])
            self.assertEqual(expected_lines, actual_lines)
            status_message = 'Setting lifecycle configuration'
        self.assertEqual(stderr.count(status_message), 2)
    _Check1()
    stdout = self.RunGsUtil(['lifecycle', 'get', suri(bucket1_uri)], return_stdout=True)
    self.assertEqual(json.loads(stdout), self.lifecycle_json_obj)
    stdout = self.RunGsUtil(['lifecycle', 'get', suri(bucket2_uri)], return_stdout=True)
    self.assertEqual(json.loads(stdout), self.lifecycle_json_obj)