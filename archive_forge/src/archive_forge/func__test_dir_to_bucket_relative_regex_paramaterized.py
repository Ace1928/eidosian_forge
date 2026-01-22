from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
import re
from unittest import mock
import six
from gslib import command
from gslib.commands import rsync
from gslib.project_id import PopulateProjectId
from gslib.storage_url import StorageUrlFromString
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForGS
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.testcase.integration_testcase import SkipForXML
from gslib.tests.util import AuthorizeProjectToUseTestingKmsKey
from gslib.tests.util import TEST_ENCRYPTION_KEY_S3
from gslib.tests.util import TEST_ENCRYPTION_KEY_S3_MD5
from gslib.tests.util import BuildErrorRegex
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import ORPHANED_FILE
from gslib.tests.util import POSIX_GID_ERROR
from gslib.tests.util import POSIX_INSUFFICIENT_ACCESS_ERROR
from gslib.tests.util import POSIX_MODE_ERROR
from gslib.tests.util import POSIX_UID_ERROR
from gslib.tests.util import SequentialAndParallelTransfer
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import SetEnvironmentForTest
from gslib.tests.util import TailSet
from gslib.tests.util import unittest
from gslib.utils.boto_util import UsingCrcmodExtension
from gslib.utils.hashing_helper import SLOW_CRCMOD_RSYNC_WARNING
from gslib.utils.posix_util import ConvertDatetimeToPOSIX
from gslib.utils.posix_util import GID_ATTR
from gslib.utils.posix_util import MODE_ATTR
from gslib.utils.posix_util import MTIME_ATTR
from gslib.utils.posix_util import NA_TIME
from gslib.utils.posix_util import UID_ATTR
from gslib.utils.retry_util import Retry
from gslib.utils.system_util import IS_OSX
from gslib.utils.system_util import IS_WINDOWS
from gslib.utils import shim_util
def _test_dir_to_bucket_relative_regex_paramaterized(self, flag, skip_dirs):
    """Test that rsync regex options work with a relative regex per the docs."""
    tmpdir = self.CreateTempDir(test_files=['a', 'b', 'c', ('data1', 'a.txt'), ('data1', 'ok'), ('data2', 'b.txt'), ('data3', 'data4', 'c.txt')])

    @Retry(AssertionError, tries=3, timeout_secs=1)
    def _check_exclude_regex(exclude_regex, expected):
        """Tests rsync skips the excluded pattern."""
        bucket_uri = self.CreateBucket()
        stderr = ''
        local = tmpdir + ('\\' if IS_WINDOWS else '/')
        stderr += self.RunGsUtil(['rsync', '-r', flag, exclude_regex, local, suri(bucket_uri)], return_stderr=True)
        listing = TailSet(tmpdir, self.FlatListDir(tmpdir))
        self.assertEqual(listing, set(['/a', '/b', '/c', '/data1/a.txt', '/data1/ok', '/data2/b.txt', '/data3/data4/c.txt']))
        actual = TailSet(suri(bucket_uri), self.FlatListBucket(bucket_uri))
        self.assertEqual(actual, expected)
        return stderr

    def _Check1():
        """Ensure the example exclude pattern from the docs works as expected."""
        _check_exclude_regex('data.[/\\\\].*\\.txt$', set(['/a', '/b', '/c', '/data1/ok']))
    _Check1()

    def _Check2():
        """Tests that a regex with a pipe works as expected."""
        _check_exclude_regex('^data|[bc]$', set(['/a']))
    _Check2()

    def _Check3():
        """Tests that directories are skipped from iteration as expected."""
        stderr = _check_exclude_regex('data3', set(['/a', '/b', '/c', '/data1/ok', '/data1/a.txt', '/data2/b.txt']))
        self.assertIn('Skipping excluded directory {}...'.format(os.path.join(tmpdir, 'data3')), stderr)
        self.assertNotIn('Skipping excluded directory {}...'.format(os.path.join(tmpdir, 'data3', 'data4')), stderr)
    if skip_dirs:
        _Check3()