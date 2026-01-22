from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import logging
import os
import re
import unittest
from boto.storage_uri import BucketStorageUri
from gslib.cs_api_map import ApiSelector
from gslib.discard_messages_queue import DiscardMessagesQueue
from gslib.gcs_json_api import GcsJsonApi
from gslib.project_id import PopulateProjectId
from gslib.tests.rewrite_helper import EnsureRewriteRestartCallbackHandler
from gslib.tests.rewrite_helper import EnsureRewriteResumeCallbackHandler
from gslib.tests.rewrite_helper import HaltingRewriteCallbackHandler
from gslib.tests.rewrite_helper import RewriteHaltException
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.util import AuthorizeProjectToUseTestingKmsKey
from gslib.tests.util import GenerationFromURI as urigen
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import TEST_ENCRYPTION_KEY1
from gslib.tests.util import TEST_ENCRYPTION_KEY2
from gslib.tests.util import TEST_ENCRYPTION_KEY3
from gslib.tests.util import TEST_ENCRYPTION_KEY4
from gslib.tests.util import unittest
from gslib.tracker_file import DeleteTrackerFile
from gslib.tracker_file import GetRewriteTrackerFilePath
from gslib.utils.encryption_helper import CryptoKeyWrapperFromKey
from gslib.utils.unit_util import ONE_MIB
def _test_rewrite_key_rotation_bucket(self, bucket_uri, command_args):
    """Helper function for testing key rotation on a bucket.

    Args:
      bucket_uri: bucket StorageUri to use for the test.
      command_args: list of args to gsutil command.
    """
    object_contents = b'bar'
    object_uri1 = self.CreateObject(bucket_uri=bucket_uri, object_name='foo/foo', contents=object_contents, encryption_key=TEST_ENCRYPTION_KEY1)
    object_uri2 = self.CreateObject(bucket_uri=bucket_uri, object_name='foo/bar', contents=object_contents, encryption_key=TEST_ENCRYPTION_KEY2)
    object_uri3 = self.CreateObject(bucket_uri=bucket_uri, object_name='foo/baz', contents=object_contents, encryption_key=TEST_ENCRYPTION_KEY3)
    object_uri4 = self.CreateObject(bucket_uri=bucket_uri, object_name='foo/qux', contents=object_contents)
    boto_config_for_test = [('GSUtil', 'encryption_key', TEST_ENCRYPTION_KEY1.decode('utf-8')), ('GSUtil', 'decryption_key1', TEST_ENCRYPTION_KEY2.decode('utf-8')), ('GSUtil', 'decryption_key2', TEST_ENCRYPTION_KEY3.decode('utf-8'))]
    with SetBotoConfigForTest(boto_config_for_test):
        stderr = self.RunGsUtil(command_args, return_stderr=True)
        self.assertIn('{} {}'.format(self.skipping_message, suri(object_uri1)), stderr)
        self.assertIn(self.rotating_message, stderr)
    for object_uri_str in (suri(object_uri1), suri(object_uri2), suri(object_uri3), suri(object_uri4)):
        self.AssertObjectUsesCSEK(object_uri_str, TEST_ENCRYPTION_KEY1)
    boto_config_for_test2 = [('GSUtil', 'decryption_key1', TEST_ENCRYPTION_KEY1)]
    with SetBotoConfigForTest(boto_config_for_test2):
        stderr = self.RunGsUtil(command_args, return_stderr=True)
        self.assertIn(self.decrypting_message, stderr)
    for object_uri_str in (suri(object_uri1), suri(object_uri2), suri(object_uri3), suri(object_uri4)):
        self.AssertObjectUnencrypted(object_uri_str)