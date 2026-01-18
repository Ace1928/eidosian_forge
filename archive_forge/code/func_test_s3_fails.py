from __future__ import absolute_import
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForGS
from gslib.tests.testcase.integration_testcase import SkipForJSON
from gslib.tests.testcase.integration_testcase import SkipForXML
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import SetBotoConfigForTest
@SkipForGS('Testing S3 only behavior')
def test_s3_fails(self):
    bucket_uri = self.CreateBucket()
    stderr = self.RunGsUtil(self._set_pap_cmd + ['inherited', suri(bucket_uri)], return_stderr=True, expected_status=1)
    if self._use_gcloud_storage:
        self.assertIn('Flags disallowed for S3', stderr)
    else:
        self.assertIn('command can only be used for GCS Buckets', stderr)
    if not self._use_gcloud_storage:
        stderr = self.RunGsUtil(self._get_pap_cmd + [suri(bucket_uri)], return_stderr=True, expected_status=1)
        self.assertIn('command can only be used for GCS Buckets', stderr)