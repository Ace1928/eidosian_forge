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
def test_setmeta(self):
    req_pays_obj_uri = self.CreateObject(bucket_uri=self.requester_pays_bucket_uri, contents=b'<html><body>text</body></html>')
    self._run_requester_pays_test(['setmeta', '-h', 'content-type:text/html', suri(req_pays_obj_uri)])
    obj_uri = self.CreateObject(bucket_uri=self.non_requester_pays_bucket_uri, contents=b'<html><body>text</body></html>')
    self._run_non_requester_pays_test(['setmeta', '-h', 'content-type:text/html', suri(obj_uri)])