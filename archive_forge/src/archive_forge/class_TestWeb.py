from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import json
import os
from unittest import mock
from gslib.commands import web
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import SetEnvironmentForTest
from gslib.utils import shim_util
@SkipForS3('Web set not supported for S3, web get returns XML.')
class TestWeb(testcase.GsUtilIntegrationTestCase):
    """Integration tests for the web command."""
    _set_web_cmd = ['web', 'set']
    _get_web_cmd = ['web', 'get']

    def test_full(self):
        bucket_uri = self.CreateBucket()
        self.RunGsUtil(self._set_web_cmd + ['-m', 'main', '-e', '404', suri(bucket_uri)])
        stdout = self.RunGsUtil(self._get_web_cmd + [suri(bucket_uri)], return_stdout=True)
        if self._use_gcloud_storage:
            self.assertIn('"mainPageSuffix": "main"', stdout)
            self.assertIn('"notFoundPage": "404"', stdout)
        else:
            self.assertEqual(json.loads(stdout), WEBCFG_FULL)

    def test_main(self):
        bucket_uri = self.CreateBucket()
        self.RunGsUtil(self._set_web_cmd + ['-m', 'main', suri(bucket_uri)])
        stdout = self.RunGsUtil(self._get_web_cmd + [suri(bucket_uri)], return_stdout=True)
        if self._use_gcloud_storage:
            self.assertEqual('{\n  "mainPageSuffix": "main"\n}\n', stdout)
        else:
            self.assertEqual(json.loads(stdout), WEBCFG_MAIN)

    def test_error(self):
        bucket_uri = self.CreateBucket()
        self.RunGsUtil(self._set_web_cmd + ['-e', '404', suri(bucket_uri)])
        stdout = self.RunGsUtil(self._get_web_cmd + [suri(bucket_uri)], return_stdout=True)
        if self._use_gcloud_storage:
            self.assertEqual('{\n  "notFoundPage": "404"\n}\n', stdout)
        else:
            self.assertEqual(json.loads(stdout), WEBCFG_ERROR)

    def test_empty(self):
        bucket_uri = self.CreateBucket()
        self.RunGsUtil(self._set_web_cmd + [suri(bucket_uri)])
        stdout = self.RunGsUtil(self._get_web_cmd + [suri(bucket_uri)], return_stdout=True)
        if self._use_gcloud_storage:
            self.assertEqual('[]\n', stdout)
        else:
            self.assertIn(WEBCFG_EMPTY, stdout)

    def testTooFewArgumentsFails(self):
        """Ensures web commands fail with too few arguments."""
        stderr = self.RunGsUtil(self._get_web_cmd, return_stderr=True, expected_status=1)
        self.assertIn('command requires at least', stderr)
        stderr = self.RunGsUtil(self._set_web_cmd, return_stderr=True, expected_status=1)
        self.assertIn('command requires at least', stderr)
        stderr = self.RunGsUtil(['web'], return_stderr=True, expected_status=1)
        self.assertIn('command requires at least', stderr)