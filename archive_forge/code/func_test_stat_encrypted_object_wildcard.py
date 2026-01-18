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
def test_stat_encrypted_object_wildcard(self):
    """Tests stat command with a mix of encrypted and unencrypted objects."""
    if self.test_api == ApiSelector.XML:
        return unittest.skip('gsutil does not support encryption with the XML API')
    bucket_uri = self.CreateBucket()
    object1_uri = self.CreateObject(bucket_uri=bucket_uri, object_name='foo1', contents=TEST_ENCRYPTION_CONTENT1, encryption_key=TEST_ENCRYPTION_KEY1)
    object2_uri = self.CreateObject(bucket_uri=bucket_uri, object_name='foo2', contents=TEST_ENCRYPTION_CONTENT2, encryption_key=TEST_ENCRYPTION_KEY2)
    object3_uri = self.CreateObject(bucket_uri=bucket_uri, object_name='foo3', contents=TEST_ENCRYPTION_CONTENT3)
    stat_string = suri(object1_uri)[:-2] + '*'
    with SetBotoConfigForTest([('GSUtil', 'encryption_key', TEST_ENCRYPTION_KEY1)]):

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _StatExpectMixed():
            """Runs stat and validates output."""
            stdout, stderr = self.RunGsUtil(['stat', stat_string], return_stdout=True, return_stderr=True)
            self.assertIn(suri(object1_uri), stdout)
            self.assertIn(TEST_ENCRYPTION_CONTENT1_MD5, stdout)
            self.assertIn(TEST_ENCRYPTION_CONTENT1_CRC32C, stdout)
            self.assertIn(TEST_ENCRYPTION_KEY1_SHA256_B64.decode('ascii'), stdout)
            self.assertIn(suri(object2_uri), stdout)
            self.assertNotIn(TEST_ENCRYPTION_CONTENT2_MD5, stdout)
            self.assertNotIn(TEST_ENCRYPTION_CONTENT2_CRC32C, stdout)
            self.assertIn('encrypted', stdout)
            self.assertIn(TEST_ENCRYPTION_KEY2_SHA256_B64.decode('ascii'), stdout)
            self.assertIn(suri(object3_uri), stdout)
            self.assertIn(TEST_ENCRYPTION_CONTENT3_MD5, stdout)
            self.assertIn(TEST_ENCRYPTION_CONTENT3_CRC32C, stdout)
        _StatExpectMixed()