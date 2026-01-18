from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import re
import gslib.tests.testcase as testcase
from gslib.project_id import PopulateProjectId
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.testcase.integration_testcase import SkipForXML
from gslib.tests.util import ObjectToURI as suri
from gslib.utils.retry_util import Retry
from gslib.utils.constants import UTF8
def test_rb(self):
    rp_bucket_uri = self.CreateBucket()
    self._set_requester_pays(rp_bucket_uri)
    self._run_requester_pays_test(['rb', suri(rp_bucket_uri)])
    non_rp_bucket_uri = self.CreateBucket()
    self._run_non_requester_pays_test(['rb', suri(non_rp_bucket_uri)])