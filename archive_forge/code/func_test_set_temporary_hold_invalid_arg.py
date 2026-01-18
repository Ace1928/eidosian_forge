from __future__ import absolute_import
import datetime
import re
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.testcase.integration_testcase import SkipForXML
from gslib.tests.util import ObjectToURI as suri
@SkipForS3('Retention is not supported for s3 objects')
@SkipForXML('Retention is not supported for XML API')
def test_set_temporary_hold_invalid_arg(self):
    object_uri = self.CreateObject()
    stderr = self.RunGsUtil(['retention', 'temp', 'held', suri(object_uri)], expected_status=1, return_stderr=True)
    self.assertRegex(stderr, 'Invalid subcommand ".*" for the "retention temp" command')