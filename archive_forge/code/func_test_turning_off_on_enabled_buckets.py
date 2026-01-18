from __future__ import absolute_import
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForGS
from gslib.tests.testcase.integration_testcase import SkipForJSON
from gslib.tests.testcase.integration_testcase import SkipForXML
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import SetBotoConfigForTest
@SkipForXML('Public access prevention only runs on GCS JSON API')
def test_turning_off_on_enabled_buckets(self):
    bucket_uri = self.CreateBucket(public_access_prevention='enforced', prefer_json_api=True)
    self.VerifyPublicAccessPreventionValue(bucket_uri, 'enforced')
    self.RunGsUtil(self._set_pap_cmd + ['inherited', suri(bucket_uri)])
    self.VerifyPublicAccessPreventionValue(bucket_uri, 'inherited')