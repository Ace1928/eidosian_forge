from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from gslib.cs_api_map import ApiSelector
from gslib.exception import NO_URLS_MATCHED_TARGET
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.testcase.integration_testcase import SkipForXML
from gslib.tests.util import GenerationFromURI as urigen
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import TEST_ENCRYPTION_CONTENT1
from gslib.tests.util import TEST_ENCRYPTION_CONTENT1_CRC32C
from gslib.tests.util import TEST_ENCRYPTION_CONTENT1_MD5
from gslib.tests.util import TEST_ENCRYPTION_CONTENT2
from gslib.tests.util import TEST_ENCRYPTION_CONTENT2_CRC32C
from gslib.tests.util import TEST_ENCRYPTION_CONTENT2_MD5
from gslib.tests.util import TEST_ENCRYPTION_CONTENT3
from gslib.tests.util import TEST_ENCRYPTION_CONTENT3_CRC32C
from gslib.tests.util import TEST_ENCRYPTION_CONTENT3_MD5
from gslib.tests.util import TEST_ENCRYPTION_KEY1
from gslib.tests.util import TEST_ENCRYPTION_KEY1_SHA256_B64
from gslib.tests.util import TEST_ENCRYPTION_KEY2
from gslib.tests.util import TEST_ENCRYPTION_KEY2_SHA256_B64
from gslib.tests.util import unittest
from gslib.utils.retry_util import Retry
def test_stat_output(self):
    """Tests stat output of a single object."""
    object_uri = self.CreateObject(contents=b'z')
    stdout = self.RunGsUtil(['stat', suri(object_uri)], return_stdout=True)
    self.assertIn(object_uri.uri, stdout)
    self.assertIn('Creation time:', stdout)
    if self.default_provider == 'gs':
        if self.test_api == ApiSelector.XML:
            self.assertIn('Cache-Control:', stdout)
            self.assertIn('Content-Encoding:', stdout)
        elif self.test_api == ApiSelector.JSON:
            self.assertIn('Storage class:', stdout)
        self.assertIn('Generation:', stdout)
        self.assertIn('Metageneration:', stdout)
        self.assertIn('Hash (crc32c):', stdout)
        self.assertIn('Hash (md5):', stdout)
        self.assertNotIn('Archived time', stdout)
    self.assertIn('Content-Length:', stdout)
    self.assertIn('Content-Type:', stdout)
    self.assertIn('ETag:', stdout)