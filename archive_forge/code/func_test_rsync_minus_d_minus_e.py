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
@unittest.skipIf(IS_WINDOWS, 'os.symlink() is not available on Windows.')
def test_rsync_minus_d_minus_e(self):
    """Tests that rsync -e ignores symlinks."""
    tmpdir = self.CreateTempDir()
    subdir = os.path.join(tmpdir, 'subdir')
    os.mkdir(subdir)
    bucket_uri = self.CreateBucket()
    fpath1 = self.CreateTempFile(tmpdir=tmpdir, file_name='obj1', contents=b'obj1')
    self.CreateTempFile(tmpdir=tmpdir, file_name='.obj2', contents=b'.obj2')
    self.CreateTempFile(tmpdir=subdir, file_name='obj3', contents=b'subdir/obj3')
    good_symlink_path = os.path.join(tmpdir, 'symlink1')
    os.symlink(fpath1, good_symlink_path)
    bad_symlink_path = os.path.join(tmpdir, 'symlink2')
    os.symlink(os.path.join('/', 'non-existent'), bad_symlink_path)
    self.CreateObject(bucket_uri=bucket_uri, object_name='.obj2', contents=b'.OBJ2')
    self.CreateObject(bucket_uri=bucket_uri, object_name='obj4', contents=b'obj4')
    self.CreateObject(bucket_uri=bucket_uri, object_name='subdir/obj5', contents=b'subdir/obj5')

    @Retry(AssertionError, tries=3, timeout_secs=1)
    def _Check1():
        """Ensure listings match the commented expectations."""
        self.RunGsUtil(['rsync', '-d', '-e', tmpdir, suri(bucket_uri)])
        listing1 = TailSet(tmpdir, self.FlatListDir(tmpdir))
        listing2 = TailSet(suri(bucket_uri), self.FlatListBucket(bucket_uri))
        self.assertEqual(listing1, set(['/obj1', '/.obj2', '/subdir/obj3', '/symlink1', '/symlink2']))
        self.assertEqual(listing2, set(['/obj1', '/.obj2', '/subdir/obj5']))
    _Check1()
    os.unlink(bad_symlink_path)

    @Retry(AssertionError, tries=3, timeout_secs=1)
    def _Check2():
        """Tests rsync works as expected."""
        self.RunGsUtil(['rsync', '-d', tmpdir, suri(bucket_uri)])
        listing1 = TailSet(tmpdir, self.FlatListDir(tmpdir))
        listing2 = TailSet(suri(bucket_uri), self.FlatListBucket(bucket_uri))
        self.assertEqual(listing1, set(['/obj1', '/.obj2', '/subdir/obj3', '/symlink1']))
        self.assertEqual(listing2, set(['/obj1', '/.obj2', '/subdir/obj5', '/symlink1']))
        self.assertEqual('obj1', self.RunGsUtil(['cat', suri(bucket_uri, 'symlink1')], return_stdout=True))
    _Check2()

    @Retry(AssertionError, tries=3, timeout_secs=1)
    def _Check3():
        stderr = self.RunGsUtil(['rsync', '-d', tmpdir, suri(bucket_uri)], return_stderr=True)
        self._VerifyNoChanges(stderr)
    _Check3()