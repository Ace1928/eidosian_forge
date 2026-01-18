from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from boto import config
from gslib.gcs_json_api import DEFAULT_HOST
from gslib.tests import testcase
from gslib.tests.testcase import integration_testcase
from gslib.tests.util import ObjectToURI
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import unittest
@integration_testcase.SkipForXML('JSON test.')
@integration_testcase.SkipForS3('Custom endpoints not available for S3.')
def test_persists_custom_endpoint_through_json_sliced_download(self):
    gs_host = config.get('Credentials', 'gs_json_host', DEFAULT_HOST)
    if gs_host == DEFAULT_HOST:
        return
    temporary_directory = self.CreateTempDir()
    with SetBotoConfigForTest([('GSUtil', 'sliced_object_download_threshold', '1B'), ('GSUtil', 'sliced_object_download_component_size', '1B')]):
        bucket_uri = self.CreateBucket()
        key_uri = self.CreateObject(bucket_uri=bucket_uri, contents=b'foo')
        stdout = self.RunGsUtil(['-DD', 'cp', ObjectToURI(key_uri), temporary_directory], env_vars=PYTHON_UNBUFFERED_ENV_VAR, return_stdout=True)
    self.assertIn(gs_host, stdout)
    self.assertNotIn(DEFAULT_HOST, stdout)