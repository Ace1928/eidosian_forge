from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import json
import posixpath
from xml.dom.minidom import parseString
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.util import ObjectToURI as suri
from gslib.utils.constants import UTF8
from gslib.utils.retry_util import Retry
from gslib.utils.translation_helper import CorsTranslation
def test_set_wildcard_non_null_cors(self):
    """Tests setting CORS on a wildcarded bucket URI."""
    random_prefix = self.MakeRandomTestString()
    bucket1_name = self.MakeTempName('bucket', prefix=random_prefix)
    bucket2_name = self.MakeTempName('bucket', prefix=random_prefix)
    bucket1_uri = self.CreateBucket(bucket_name=bucket1_name)
    bucket2_uri = self.CreateBucket(bucket_name=bucket2_name)
    common_prefix = posixpath.commonprefix([suri(bucket1_uri), suri(bucket2_uri)])
    self.assertTrue(common_prefix.startswith('gs://%sgsutil-test-test-set-wildcard-non' % random_prefix))
    wildcard = '%s*' % common_prefix
    fpath = self.CreateTempFile(contents=self.cors_doc.encode(UTF8))
    if self._use_gcloud_storage:
        expected = set(['Updating %s' % suri(bucket1_uri), 'Updating %s' % suri(bucket2_uri)])
    else:
        expected = set(['Setting CORS on %s/...' % suri(bucket1_uri), 'Setting CORS on %s/...' % suri(bucket2_uri)])
    actual = set()

    @Retry(AssertionError, tries=3, timeout_secs=1)
    def _Check1():
        """Ensures expect set lines are present in command output."""
        stderr = self.RunGsUtil(self._set_cmd_prefix + [fpath, wildcard], return_stderr=True)
        outlines = stderr.splitlines()
        for line in outlines:
            if 'You are using a deprecated alias' in line or 'gsutil help cors' in line or 'Please use "cors" with the appropriate sub-command' in line:
                continue
            actual.add(line)
        for line in expected:
            if self._use_gcloud_storage:
                self.assertIn(line, stderr)
            else:
                self.assertIn(line, actual)
                self.assertEqual(stderr.count('Setting CORS'), 2)
    _Check1()
    stdout = self.RunGsUtil(self._get_cmd_prefix + [suri(bucket1_uri)], return_stdout=True)
    self.assertEqual(json.loads(stdout), self.cors_json_obj)
    stdout = self.RunGsUtil(self._get_cmd_prefix + [suri(bucket2_uri)], return_stdout=True)
    self.assertEqual(json.loads(stdout), self.cors_json_obj)