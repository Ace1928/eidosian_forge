from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from datetime import datetime
import os
import posixpath
import re
import stat
import subprocess
import sys
import time
import gslib
from gslib.commands import ls
from gslib.cs_api_map import ApiSelector
from gslib.project_id import PopulateProjectId
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForGS
from gslib.tests.testcase.integration_testcase import SkipForJSON
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.testcase.integration_testcase import SkipForXML
from gslib.tests.util import CaptureStdout
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import RUN_S3_TESTS
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import SetEnvironmentForTest
from gslib.tests.util import TEST_ENCRYPTION_CONTENT1
from gslib.tests.util import TEST_ENCRYPTION_CONTENT1_CRC32C
from gslib.tests.util import TEST_ENCRYPTION_CONTENT1_MD5
from gslib.tests.util import TEST_ENCRYPTION_CONTENT2
from gslib.tests.util import TEST_ENCRYPTION_CONTENT2_CRC32C
from gslib.tests.util import TEST_ENCRYPTION_CONTENT2_MD5
from gslib.tests.util import TEST_ENCRYPTION_CONTENT3
from gslib.tests.util import TEST_ENCRYPTION_CONTENT3_CRC32C
from gslib.tests.util import TEST_ENCRYPTION_CONTENT3_MD5
from gslib.tests.util import TEST_ENCRYPTION_CONTENT4
from gslib.tests.util import TEST_ENCRYPTION_CONTENT4_CRC32C
from gslib.tests.util import TEST_ENCRYPTION_CONTENT4_MD5
from gslib.tests.util import TEST_ENCRYPTION_CONTENT5
from gslib.tests.util import TEST_ENCRYPTION_CONTENT5_CRC32C
from gslib.tests.util import TEST_ENCRYPTION_CONTENT5_MD5
from gslib.tests.util import TEST_ENCRYPTION_KEY1
from gslib.tests.util import TEST_ENCRYPTION_KEY1_SHA256_B64
from gslib.tests.util import TEST_ENCRYPTION_KEY2
from gslib.tests.util import TEST_ENCRYPTION_KEY2_SHA256_B64
from gslib.tests.util import TEST_ENCRYPTION_KEY3
from gslib.tests.util import TEST_ENCRYPTION_KEY3_SHA256_B64
from gslib.tests.util import TEST_ENCRYPTION_KEY4
from gslib.tests.util import TEST_ENCRYPTION_KEY4_SHA256_B64
from gslib.tests.util import unittest
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
from gslib.utils.constants import UTF8
from gslib.utils.ls_helper import PrintFullInfoAboutObject
from gslib.utils.retry_util import Retry
from gslib.utils.system_util import IS_WINDOWS
from gslib.utils import shim_util
from six import add_move, MovedModule
from six.moves import mock
def test_one_object_with_L_storage_class_update(self):
    """Tests the JSON storage class update time field."""
    if self.test_api == ApiSelector.XML:
        return unittest.skip('XML API has no concept of storage class update time')
    current_time = datetime(2017, 1, 2, 3, 4, 5, 6, tzinfo=None)
    obj_metadata = apitools_messages.Object(name='foo', bucket='bar', timeCreated=current_time, updated=current_time, timeStorageClassUpdated=current_time, etag='12345')
    obj_ref = mock.Mock()
    obj_ref.root_object = obj_metadata
    obj_ref.url_string = 'foo'
    with CaptureStdout() as output:
        PrintFullInfoAboutObject(obj_ref)
    output = '\n'.join(output)
    find_stor_update_re = re.compile('^\\s*Storage class update time:+(?P<stor_update_time_val>.+)$', re.MULTILINE)
    stor_update_time_match = re.search(find_stor_update_re, output)
    self.assertIsNone(stor_update_time_match)
    new_update_time = datetime(2017, 2, 3, 4, 5, 6, 7, tzinfo=None)
    obj_metadata2 = apitools_messages.Object(name='foo2', bucket='bar2', timeCreated=current_time, updated=current_time, timeStorageClassUpdated=new_update_time, etag='12345')
    obj_ref2 = mock.Mock()
    obj_ref2.root_object = obj_metadata2
    obj_ref2.url_string = 'foo2'
    with CaptureStdout() as output2:
        PrintFullInfoAboutObject(obj_ref2)
    output2 = '\n'.join(output2)
    find_time_created_re = re.compile('^\\s*Creation time:\\s+(?P<time_created_val>.+)$', re.MULTILINE)
    time_created_match = re.search(find_time_created_re, output2)
    self.assertIsNotNone(time_created_match)
    time_created = time_created_match.group('time_created_val')
    self.assertEqual(time_created, datetime.strftime(current_time, '%a, %d %b %Y %H:%M:%S GMT'))
    find_stor_update_re_2 = re.compile('^\\s*Storage class update time:+(?P<stor_update_time_val_2>.+)$', re.MULTILINE)
    stor_update_time_match_2 = re.search(find_stor_update_re_2, output2)
    self.assertIsNotNone(stor_update_time_match_2)
    stor_update_time = stor_update_time_match_2.group('stor_update_time_val_2')
    self.assertEqual(stor_update_time, datetime.strftime(new_update_time, '%a, %d %b %Y %H:%M:%S GMT'))