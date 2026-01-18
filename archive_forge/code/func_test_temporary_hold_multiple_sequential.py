from __future__ import absolute_import
import datetime
import re
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.testcase.integration_testcase import SkipForXML
from gslib.tests.util import ObjectToURI as suri
@SkipForS3('Retention is not supported for s3 objects')
@SkipForXML('Retention is not supported for XML API')
def test_temporary_hold_multiple_sequential(self):
    bucket_uri = self.CreateBucket()
    object1_uri = self.CreateObject(bucket_uri=bucket_uri, contents='content')
    object2_uri = self.CreateObject(bucket_uri=bucket_uri, contents='content')
    self.RunGsUtil(['retention', 'temp', 'set', suri(object1_uri), suri(object2_uri)])
    self._VerifyObjectHoldAndRetentionStatus(bucket_uri, object1_uri, temporary_hold=True)
    self._VerifyObjectHoldAndRetentionStatus(bucket_uri, object2_uri, temporary_hold=True)
    self.RunGsUtil(['retention', 'temp', 'release', suri(object1_uri), suri(object2_uri)])
    self._VerifyObjectHoldAndRetentionStatus(bucket_uri, object1_uri, temporary_hold=False)
    self._VerifyObjectHoldAndRetentionStatus(bucket_uri, object2_uri, temporary_hold=False)