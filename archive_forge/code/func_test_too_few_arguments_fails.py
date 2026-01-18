from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import re
from unittest import skipIf
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.testcase.integration_testcase import SkipForXML
from gslib.tests.util import ObjectToURI as suri
def test_too_few_arguments_fails(self):
    stderr = self.RunGsUtil(self._set_dsc_cmd, return_stderr=True, expected_status=1)
    self.assertIn('command requires at least', stderr)
    if self._use_gcloud_storage:
        expected_status = 2
        expected_error_string = 'argument URL [URL ...]: Must be specified'
    else:
        expected_status = 1
        expected_error_string = 'command requires at least'
    stderr = self.RunGsUtil(self._set_dsc_cmd + ['std'], return_stderr=True, expected_status=expected_status)
    self.assertIn(expected_error_string, stderr)
    stderr = self.RunGsUtil(self._get_dsc_cmd, return_stderr=True, expected_status=1)
    self.assertIn('command requires at least', stderr)