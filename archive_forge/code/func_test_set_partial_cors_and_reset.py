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
def test_set_partial_cors_and_reset(self):
    """Tests setting CORS without maxAgeSeconds, then removing it."""
    bucket_uri = self.CreateBucket()
    tmpdir = self.CreateTempDir()
    fpath = self.CreateTempFile(tmpdir=tmpdir, contents=self.cors_doc2)
    self.RunGsUtil(self._set_cmd_prefix + [fpath, suri(bucket_uri)])
    stdout = self.RunGsUtil(self._get_cmd_prefix + [suri(bucket_uri)], return_stdout=True)
    self.assertEqual(json.loads(stdout), self.cors_json_obj2)
    fpath = self.CreateTempFile(tmpdir=tmpdir, contents=self.empty_doc1)
    self.RunGsUtil(self._set_cmd_prefix + [fpath, suri(bucket_uri)])
    stdout = self.RunGsUtil(self._get_cmd_prefix + [suri(bucket_uri)], return_stdout=True)
    self.assertIn(self.no_cors, stdout)