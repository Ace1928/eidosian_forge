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
class TestRsync(testcase.GsUtilIntegrationTestCase):
    """Integration tests for rsync command."""

    def _GetMetadataAttribute(self, bucket_name, object_name, attr_name):
        """Retrieves and returns an attribute from an objects metadata.

    Args:
      bucket_name: The name of the bucket the object is in.
      object_name: The name of the object itself.
      attr_name: The name of the custom metadata attribute.

    Returns:
      The value at the specified attribute name in the metadata. If not present,
      returns None.
    """
        gsutil_api = self.json_api if self.default_provider == 'gs' else self.xml_api
        metadata = gsutil_api.GetObjectMetadata(bucket_name, object_name, provider=self.default_provider, fields=[attr_name])
        return getattr(metadata, attr_name, None)

    def _VerifyNoChanges(self, stderr):
        if self._use_gcloud_storage:
            self.assertNotIn('Completed', stderr)
        else:
            self.assertEqual(NO_CHANGES, stderr)

    def _VerifyObjectMtime(self, bucket_name, object_name, expected_mtime, expected_present=True):
        """Retrieves the object's mtime.

    Args:
      bucket_name: The name of the bucket the object is in.
      object_name: The name of the object itself.
      expected_mtime: The expected retrieved mtime.
      expected_present: True if the mtime must be present in the
          object metadata, False if it must not be present.


    Returns:
      None
    """
        self.VerifyObjectCustomAttribute(bucket_name, object_name, MTIME_ATTR, expected_mtime, expected_present=expected_present)

    def test_invalid_args(self):
        """Tests various invalid argument cases."""
        bucket_uri = self.CreateBucket()
        obj1 = self.CreateObject(bucket_uri=bucket_uri, object_name='obj1', contents=b'obj1')
        tmpdir = self.CreateTempDir()
        self.RunGsUtil(['rsync', suri(obj1), suri(bucket_uri)], expected_status=1)
        self.RunGsUtil(['rsync', suri(bucket_uri), suri(obj1)], expected_status=1)
        self.RunGsUtil(['rsync', suri(bucket_uri), 'gs://' + self.nonexistent_bucket_name], expected_status=1)
        self.RunGsUtil(['rsync', suri(obj1), tmpdir], expected_status=1)
        if not self._use_gcloud_storage:
            self.RunGsUtil(['rsync', tmpdir, suri(obj1)], expected_status=1)
        self.RunGsUtil(['rsync', tmpdir, suri(obj1), 'gs://' + self.nonexistent_bucket_name], expected_status=1)

    def test_invalid_src_mtime(self):
        """Tests that an exception is thrown if mtime cannot be cast as a long."""
        bucket1_uri = self.CreateBucket()
        bucket2_uri = self.CreateBucket()
        self.CreateObject(bucket_uri=bucket1_uri, object_name='obj1', contents=b'obj1', mtime='xyz')
        self.CreateObject(bucket_uri=bucket1_uri, object_name='obj2', contents=b'obj2', mtime=123)
        self.CreateObject(bucket_uri=bucket1_uri, object_name='obj3', contents=b'obj3', mtime=long(1234567891011))
        self.CreateObject(bucket_uri=bucket1_uri, object_name='obj4', contents=b'obj4', mtime=-100)
        self.CreateObject(bucket_uri=bucket1_uri, object_name='obj5', contents=b'obj5', mtime=-1)

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check1():
            stderr = self.RunGsUtil(['rsync', suri(bucket1_uri), suri(bucket2_uri)], return_stderr=True)
            if self._use_gcloud_storage:
                self.assertRegex(stderr, 'obj1#\\d+ metadata did not contain a numeric value for goog-reserved-file-mtime')
                self.assertNotRegex(stderr, 'obj2#\\d+ metadata did not contain a numeric value for goog-reserved-file-mtime')
                self.assertRegex(stderr, 'obj3#\\d+ metadata that is more than one day in the future from the system time')
                self.assertRegex(stderr, 'Found negative time value in gs://.*/obj4')
                self.assertRegex(stderr, 'Found negative time value in gs://.*/obj5')
            else:
                self.assertIn('obj1 has an invalid mtime in its metadata', stderr)
                self.assertNotIn('obj2 has an invalid mtime in its metadata', stderr)
                self.assertIn('obj3 has an mtime more than 1 day from current system time', stderr)
                self.assertIn('obj4 has a negative mtime in its metadata', stderr)
                self.assertIn('obj5 has a negative mtime in its metadata', stderr)
        _Check1()

    @unittest.skipIf(IS_WINDOWS, 'POSIX attributes not available on Windows.')
    @unittest.skipUnless(UsingCrcmodExtension(), 'Test requires fast crcmod.')
    def test_bucket_to_bucket_preserve_posix(self):
        """Tests that rsync -P works with bucket to bucket."""
        src_bucket = self.CreateBucket()
        dst_bucket = self.CreateBucket()
        primary_gid = os.getgid()
        non_primary_gid = util.GetNonPrimaryGid()
        self.CreateObject(bucket_uri=src_bucket, object_name='obj1', contents=b'obj1', mode='444')
        self.CreateObject(bucket_uri=src_bucket, object_name='obj2', contents=b'obj2', gid=primary_gid)
        self.CreateObject(bucket_uri=src_bucket, object_name='obj3', contents=b'obj3', gid=non_primary_gid)
        self.CreateObject(bucket_uri=src_bucket, object_name='obj4', contents=b'obj3', uid=INVALID_UID(), gid=INVALID_GID(), mode='222')
        self.CreateObject(bucket_uri=src_bucket, object_name='obj5', contents=b'obj5', uid=USER_ID, gid=primary_gid, mode=str(DEFAULT_MODE))
        self.CreateObject(bucket_uri=dst_bucket, object_name='obj5', contents=b'obj5')

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check1():
            """Test bucket to bucket rsync with -P flag and verify attributes."""
            stderr = self.RunGsUtil(['rsync', '-P', suri(src_bucket), suri(dst_bucket)], return_stderr=True)
            listing1 = TailSet(suri(src_bucket), self.FlatListBucket(src_bucket))
            listing2 = TailSet(suri(dst_bucket), self.FlatListBucket(dst_bucket))
            self.assertEqual(listing1, set(['/obj1', '/obj2', '/obj3', '/obj4', '/obj5']))
            self.assertEqual(listing2, set(['/obj1', '/obj2', '/obj3', '/obj4', '/obj5']))
            if self._use_gcloud_storage:
                self.assertIn('Patching', stderr)
            else:
                self.assertIn('Copying POSIX attributes from src to dst for', stderr)
        _Check1()
        self.VerifyObjectCustomAttribute(dst_bucket.bucket_name, 'obj1', MODE_ATTR, '444')
        self.VerifyObjectCustomAttribute(dst_bucket.bucket_name, 'obj2', GID_ATTR, str(primary_gid))
        self.VerifyObjectCustomAttribute(dst_bucket.bucket_name, 'obj3', GID_ATTR, str(non_primary_gid))
        self.VerifyObjectCustomAttribute(dst_bucket.bucket_name, 'obj4', GID_ATTR, str(INVALID_GID()))
        self.VerifyObjectCustomAttribute(dst_bucket.bucket_name, 'obj4', UID_ATTR, str(INVALID_UID()))
        self.VerifyObjectCustomAttribute(dst_bucket.bucket_name, 'obj4', MODE_ATTR, '222')
        self.VerifyObjectCustomAttribute(dst_bucket.bucket_name, 'obj5', UID_ATTR, str(USER_ID))
        self.VerifyObjectCustomAttribute(dst_bucket.bucket_name, 'obj5', GID_ATTR, str(primary_gid))
        self.VerifyObjectCustomAttribute(dst_bucket.bucket_name, 'obj5', MODE_ATTR, str(DEFAULT_MODE))

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check2():
            """Check that we are not patching destination metadata a second time."""
            stderr = self.RunGsUtil(['rsync', '-P', suri(src_bucket), suri(dst_bucket)], return_stderr=True)
            if self._use_gcloud_storage:
                self.assertNotIn('Patching', stderr)
            else:
                self.assertNotIn('Copying POSIX attributes from src to dst for', stderr)
        _Check2()

    def test_bucket_to_bucket_same_objects_src_mtime(self):
        """Tests bucket to bucket with mtime.

    Each has the same items but only the source has mtime stored in its
    metadata.
    Ensure that destination now also has the mtime of the files in its metadata.
    """
        src_bucket = self.CreateBucket()
        dst_bucket = self.CreateBucket()
        self.CreateObject(bucket_uri=src_bucket, object_name='obj1', contents=b'obj1', mtime=0)
        self.CreateObject(bucket_uri=src_bucket, object_name='subdir/obj2', contents=b'subdir/obj2', mtime=1)
        self.CreateObject(bucket_uri=dst_bucket, object_name='obj1', contents=b'obj1')
        self.CreateObject(bucket_uri=dst_bucket, object_name='subdir/obj2', contents=b'subdir/obj2')

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check1():
            """Tests rsync works as expected."""
            self.RunGsUtil(['rsync', '-r', suri(src_bucket), suri(dst_bucket)])
            listing1 = TailSet(suri(src_bucket), self.FlatListBucket(src_bucket))
            self.assertEqual(listing1, set(['/obj1', '/subdir/obj2']))
        _Check1()

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check2():
            stderr = self.RunGsUtil(['rsync', suri(src_bucket), suri(dst_bucket)], return_stderr=True)
            self._VerifyNoChanges(stderr)
        _Check2()
        if self._use_gcloud_storage:
            self._VerifyObjectMtime(dst_bucket.bucket_name, 'obj1', NA_TIME, expected_present=False)
            self._VerifyObjectMtime(dst_bucket.bucket_name, 'subdir/obj2', NA_TIME, expected_present=False)
        else:
            self._VerifyObjectMtime(dst_bucket.bucket_name, 'obj1', '0')
            self._VerifyObjectMtime(dst_bucket.bucket_name, 'subdir/obj2', '1')

    def test_bucket_to_bucket_src_mtime(self):
        """Tests bucket to bucket where source has mtime in files."""
        src_bucket = self.CreateBucket()
        dst_bucket = self.CreateBucket()
        obj1 = self.CreateObject(bucket_uri=src_bucket, object_name='obj1', contents=b'obj1', mtime=0)
        obj2 = self.CreateObject(bucket_uri=src_bucket, object_name='subdir/obj2', contents=b'subdir/obj2', mtime=1)
        self._VerifyObjectMtime(obj1.bucket_name, obj1.object_name, '0')
        self._VerifyObjectMtime(obj2.bucket_name, obj2.object_name, '1')

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check1():
            """Tests rsync works as expected."""
            self.RunGsUtil(['rsync', '-r', suri(src_bucket), suri(dst_bucket)])
            listing1 = TailSet(suri(src_bucket), self.FlatListBucket(src_bucket))
            listing2 = TailSet(suri(dst_bucket), self.FlatListBucket(dst_bucket))
            self.assertEqual(listing1, set(['/obj1', '/subdir/obj2']))
            self.assertEqual(listing2, set(['/obj1', '/subdir/obj2']))
        _Check1()
        self._VerifyObjectMtime(dst_bucket.bucket_name, 'obj1', '0')
        self._VerifyObjectMtime(dst_bucket.bucket_name, 'subdir/obj2', '1')

    def test_bucket_to_bucket_dst_mtime(self):
        """Tests bucket to bucket where destination has mtime in objects."""
        src_bucket = self.CreateBucket()
        dst_bucket = self.CreateBucket()
        self.CreateObject(bucket_uri=src_bucket, object_name='obj1', contents=b'OBJ1')
        self.CreateObject(bucket_uri=src_bucket, object_name='subdir/obj2', contents=b'subdir/obj2')
        self.CreateObject(bucket_uri=src_bucket, object_name='.obj3', contents=b'.obj3')
        self.CreateObject(bucket_uri=src_bucket, object_name='subdir/obj4', contents=b'subdir/obj4')
        self.CreateObject(bucket_uri=src_bucket, object_name='obj6', contents=b'OBJ6', mtime=100)
        self.CreateObject(bucket_uri=dst_bucket, object_name='obj1', contents=b'obj1', mtime=10)
        self.CreateObject(bucket_uri=dst_bucket, object_name='subdir/obj2', contents=b'subdir/obj2', mtime=10)
        self.CreateObject(bucket_uri=dst_bucket, object_name='.obj3', contents=b'.OBJ3', mtime=long(1000000000000))
        self.CreateObject(bucket_uri=dst_bucket, object_name='subdir/obj5', contents=b'subdir/obj5', mtime=10)
        self.CreateObject(bucket_uri=dst_bucket, object_name='obj6', contents=b'obj6', mtime=100)

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check1():
            """Tests rsync works as expected."""
            (self.RunGsUtil(['rsync', '-r', '-d', suri(src_bucket), suri(dst_bucket)], return_stderr=True),)
            listing1 = TailSet(suri(src_bucket), self.FlatListBucket(src_bucket))
            listing2 = TailSet(suri(dst_bucket), self.FlatListBucket(dst_bucket))
            self.assertEqual(listing1, set(['/obj1', '/subdir/obj2', '/.obj3', '/subdir/obj4', '/obj6']))
            self.assertEqual(listing2, set(['/obj1', '/subdir/obj2', '/.obj3', '/subdir/obj4', '/obj6']))
        _Check1()
        self._VerifyObjectMtime(dst_bucket.bucket_name, 'subdir/obj2', '10')
        if self._use_gcloud_storage:
            source_o1_time_created = self.GetObjectMetadataWithFields(src_bucket.bucket_name, 'obj1', ['timeCreated']).timeCreated
            source_o1_posix_time_created = str(ConvertDatetimeToPOSIX(source_o1_time_created))
            self._VerifyObjectMtime(dst_bucket.bucket_name, 'obj1', source_o1_posix_time_created)
            source_o4_time_created = self.GetObjectMetadataWithFields(src_bucket.bucket_name, 'subdir/obj4', ['timeCreated']).timeCreated
            source_o4_posix_time_created = str(ConvertDatetimeToPOSIX(source_o4_time_created))
            self._VerifyObjectMtime(dst_bucket.bucket_name, 'subdir/obj4', source_o4_posix_time_created)
        else:
            self._VerifyObjectMtime(dst_bucket.bucket_name, 'obj1', NA_TIME, expected_present=False)
            self._VerifyObjectMtime(dst_bucket.bucket_name, 'subdir/obj4', NA_TIME, expected_present=False)

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check3():
            stderr = self.RunGsUtil(['rsync', suri(src_bucket), suri(dst_bucket)], return_stderr=True)
            self._VerifyNoChanges(stderr)
        _Check3()

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check4():
            self.assertEqual('OBJ1', self.RunGsUtil(['cat', suri(dst_bucket, 'obj1')], return_stdout=True))
            self.assertEqual('.obj3', self.RunGsUtil(['cat', suri(dst_bucket, '.obj3')], return_stdout=True))
            self.assertEqual('OBJ6', self.RunGsUtil(['cat', suri(dst_bucket, 'obj6')], return_stdout=True))
        _Check4()

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check5():
            """Tests rsync -c works as expected."""
            self.RunGsUtil(['rsync', '-r', '-d', '-c', suri(src_bucket), suri(dst_bucket)])
            listing1 = TailSet(suri(src_bucket), self.FlatListBucket(src_bucket))
            listing2 = TailSet(suri(dst_bucket), self.FlatListBucket(dst_bucket))
            self.assertEqual(listing1, set(['/obj1', '/subdir/obj2', '/.obj3', '/subdir/obj4', '/obj6']))
            self.assertEqual(listing2, set(['/obj1', '/subdir/obj2', '/.obj3', '/subdir/obj4', '/obj6']))
            self.assertEqual('OBJ6', self.RunGsUtil(['cat', suri(dst_bucket, 'obj6')], return_stdout=True))
            self._VerifyObjectMtime(dst_bucket.bucket_name, 'obj6', '100')
        _Check5()

    def test_bucket_to_bucket(self):
        """Tests that flat and recursive rsync between 2 buckets works correctly."""
        bucket1_uri = self.CreateBucket()
        bucket2_uri = self.CreateBucket()
        self.CreateObject(bucket_uri=bucket1_uri, object_name='obj1', contents=b'obj1')
        self.CreateObject(bucket_uri=bucket1_uri, object_name='.obj2', contents=b'.obj2', mtime=10)
        self.CreateObject(bucket_uri=bucket1_uri, object_name='subdir/obj3', contents=b'subdir/obj3')
        self.CreateObject(bucket_uri=bucket1_uri, object_name='obj6', contents=b'obj6_', mtime=100)
        self.CreateObject(bucket_uri=bucket2_uri, object_name='.obj2', contents=b'.OBJ2')
        self.CreateObject(bucket_uri=bucket2_uri, object_name='obj4', contents=b'obj4')
        self.CreateObject(bucket_uri=bucket2_uri, object_name='subdir/obj5', contents=b'subdir/obj5')
        self.CreateObject(bucket_uri=bucket2_uri, object_name='obj6', contents=b'obj6', mtime=100)

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check1():
            """Tests rsync works as expected."""
            self.RunGsUtil(['rsync', suri(bucket1_uri), suri(bucket2_uri)])
            listing1 = TailSet(suri(bucket1_uri), self.FlatListBucket(bucket1_uri))
            listing2 = TailSet(suri(bucket2_uri), self.FlatListBucket(bucket2_uri))
            self.assertEqual(listing1, set(['/obj1', '/.obj2', '/subdir/obj3', '/obj6']))
            self.assertEqual(listing2, set(['/obj1', '/.obj2', '/obj4', '/subdir/obj5', '/obj6']))
            self.assertEqual('.obj2', self.RunGsUtil(['cat', suri(bucket1_uri, '.obj2')], return_stdout=True))
            self.assertEqual('.obj2', self.RunGsUtil(['cat', suri(bucket2_uri, '.obj2')], return_stdout=True))
            self.assertEqual('obj6_', self.RunGsUtil(['cat', suri(bucket2_uri, 'obj6')], return_stdout=True))
            self._VerifyObjectMtime(bucket2_uri.bucket_name, '.obj2', '10')
        _Check1()

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check2():
            stderr = self.RunGsUtil(['rsync', suri(bucket1_uri), suri(bucket2_uri)], return_stderr=True)
            self._VerifyNoChanges(stderr)
        _Check2()
        self.CreateObject(bucket_uri=bucket1_uri, object_name='obj6', contents=b'obj6')
        self.CreateObject(bucket_uri=bucket2_uri, object_name='obj7', contents=b'obj7')
        self.RunGsUtil(['rm', suri(bucket1_uri, 'obj1')])
        self.RunGsUtil(['rm', suri(bucket2_uri, '.obj2')])

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check3():
            self.RunGsUtil(['rsync', '-r', suri(bucket1_uri), suri(bucket2_uri)])
            listing1 = TailSet(suri(bucket1_uri), self.FlatListBucket(bucket1_uri))
            listing2 = TailSet(suri(bucket2_uri), self.FlatListBucket(bucket2_uri))
            self.assertEqual(listing1, set(['/.obj2', '/obj6', '/subdir/obj3']))
            self.assertEqual(listing2, set(['/obj1', '/.obj2', '/obj4', '/obj6', '/obj7', '/subdir/obj3', '/subdir/obj5']))
        _Check3()

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check4():
            stderr = self.RunGsUtil(['rsync', '-r', suri(bucket1_uri), suri(bucket2_uri)], return_stderr=True)
            self._VerifyNoChanges(stderr)
        _Check4()

    def test_bucket_to_bucket_minus_d(self):
        """Tests that flat and recursive rsync between 2 buckets works correctly."""
        bucket1_uri = self.CreateBucket()
        bucket2_uri = self.CreateBucket()
        self.CreateObject(bucket_uri=bucket1_uri, object_name='obj1', contents=b'obj1')
        self.CreateObject(bucket_uri=bucket1_uri, object_name='.obj2', contents=b'.obj2')
        self.CreateObject(bucket_uri=bucket1_uri, object_name='subdir/obj3', contents=b'subdir/obj3')
        self.CreateObject(bucket_uri=bucket2_uri, object_name='.obj2', contents=b'.OBJ2')
        self.CreateObject(bucket_uri=bucket2_uri, object_name='obj4', contents=b'obj4')
        self.CreateObject(bucket_uri=bucket2_uri, object_name='subdir/obj5', contents=b'subdir/obj5')

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check1():
            """Tests rsync works as expected."""
            self.RunGsUtil(['rsync', '-d', suri(bucket1_uri), suri(bucket2_uri)])
            listing1 = TailSet(suri(bucket1_uri), self.FlatListBucket(bucket1_uri))
            listing2 = TailSet(suri(bucket2_uri), self.FlatListBucket(bucket2_uri))
            self.assertEqual(listing1, set(['/obj1', '/.obj2', '/subdir/obj3']))
            self.assertEqual(listing2, set(['/obj1', '/.obj2', '/subdir/obj5']))
            self.assertEqual('.obj2', self.RunGsUtil(['cat', suri(bucket1_uri, '.obj2')], return_stdout=True))
            self.assertEqual('.obj2', self.RunGsUtil(['cat', suri(bucket2_uri, '.obj2')], return_stdout=True))
        _Check1()

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check2():
            stderr = self.RunGsUtil(['rsync', '-d', suri(bucket1_uri), suri(bucket2_uri)], return_stderr=True)
            self._VerifyNoChanges(stderr)
        _Check2()
        self.CreateObject(bucket_uri=bucket1_uri, object_name='obj6', contents=b'obj6')
        self.CreateObject(bucket_uri=bucket2_uri, object_name='obj7', contents=b'obj7')
        self.RunGsUtil(['rm', suri(bucket1_uri, 'obj1')])
        self.RunGsUtil(['rm', suri(bucket2_uri, '.obj2')])

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check3():
            self.RunGsUtil(['rsync', '-d', '-r', suri(bucket1_uri), suri(bucket2_uri)])
            listing1 = TailSet(suri(bucket1_uri), self.FlatListBucket(bucket1_uri))
            listing2 = TailSet(suri(bucket2_uri), self.FlatListBucket(bucket2_uri))
            self.assertEqual(listing1, set(['/.obj2', '/obj6', '/subdir/obj3']))
            self.assertEqual(listing2, set(['/.obj2', '/obj6', '/subdir/obj3']))
        _Check3()

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check4():
            stderr = self.RunGsUtil(['rsync', '-d', '-r', suri(bucket1_uri), suri(bucket2_uri)], return_stderr=True)
            self._VerifyNoChanges(stderr)
        _Check4()

    @SequentialAndParallelTransfer
    @unittest.skipUnless(UsingCrcmodExtension(), 'Test requires fast crcmod.')
    def test_dir_to_bucket_mtime(self):
        """Tests dir to bucket with mtime.

    Each has the same items, the source has mtime for all objects, whereas dst
    only has mtime for obj5 and obj6 to test for different a later mtime at src
    and the same mtime from src to dst, respectively. Ensure that destination
    now also has the mtime of the files in its metadata.
    """
        tmpdir = self.CreateTempDir()
        subdir = os.path.join(tmpdir, 'subdir')
        os.mkdir(subdir)
        self.CreateTempFile(tmpdir=tmpdir, file_name='obj1', contents=b'obj1', mtime=10)
        self.CreateTempFile(tmpdir=tmpdir, file_name='.obj2', contents=b'.obj2', mtime=10)
        self.CreateTempFile(tmpdir=subdir, file_name='obj3', contents=b'subdir/obj3', mtime=10)
        self.CreateTempFile(tmpdir=subdir, file_name='obj5', contents=b'subdir/obj5', mtime=15)
        self.CreateTempFile(tmpdir=tmpdir, file_name='obj6', contents=b'obj6', mtime=100)
        self.CreateTempFile(tmpdir=tmpdir, file_name='obj7', contents=b'obj7_', mtime=100)
        bucket_uri = self.CreateBucket()
        self.CreateObject(bucket_uri=bucket_uri, object_name='obj1', contents=b'OBJ1')
        self.CreateObject(bucket_uri=bucket_uri, object_name='.obj2', contents=b'.obj2')
        self._SetObjectCustomMetadataAttribute(self.default_provider, bucket_uri.bucket_name, '.obj2', 'test', 'test')
        self.CreateObject(bucket_uri=bucket_uri, object_name='obj4', contents=b'obj4')
        self.CreateObject(bucket_uri=bucket_uri, object_name='subdir/obj5', contents=b'subdir/obj5', mtime=10)
        self.CreateObject(bucket_uri=bucket_uri, object_name='obj6', contents=b'OBJ6', mtime=100)
        self.CreateObject(bucket_uri=bucket_uri, object_name='obj7', contents=b'obj7', mtime=100)
        cumulative_stderr = set()

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check1():
            """Tests rsync works as expected."""
            stderr = self.RunGsUtil(['rsync', '-r', '-d', tmpdir, suri(bucket_uri)], return_stderr=True)
            cumulative_stderr.update([s for s in stderr.splitlines() if s])
            listing1 = TailSet(tmpdir, self.FlatListDir(tmpdir))
            listing2 = TailSet(suri(bucket_uri), self.FlatListBucket(bucket_uri))
            self.assertEqual(listing1, set(['/obj1', '/.obj2', '/subdir/obj3', '/subdir/obj5', '/obj6', '/obj7']))
            self.assertEqual(listing2, set(['/obj1', '/.obj2', '/subdir/obj3', '/subdir/obj5', '/obj6', '/obj7']))
            self.assertEqual('OBJ6', self.RunGsUtil(['cat', suri(bucket_uri, 'obj6')], return_stdout=True))
            self.assertEqual('obj7_', self.RunGsUtil(['cat', suri(bucket_uri, 'obj7')], return_stdout=True))
        _Check1()

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check2():
            stderr = self.RunGsUtil(['rsync', '-r', '-d', tmpdir, suri(bucket_uri)], return_stderr=True)
            self._VerifyNoChanges(stderr)
        _Check2()
        self._VerifyObjectMtime(bucket_uri.bucket_name, 'obj1', '10')
        self._VerifyObjectMtime(bucket_uri.bucket_name, '.obj2', '10')
        self._VerifyObjectMtime(bucket_uri.bucket_name, 'subdir/obj3', '10')
        self._VerifyObjectMtime(bucket_uri.bucket_name, 'subdir/obj5', '15')
        self._VerifyObjectMtime(bucket_uri.bucket_name, 'obj6', '100')
        copied_over_object_notice = "Copying whole file/object for %s instead of patching because you don't have owner permission on the object." % suri(bucket_uri, '.obj2')
        if copied_over_object_notice not in cumulative_stderr and (not self._use_gcloud_storage):
            self.VerifyObjectCustomAttribute(bucket_uri.bucket_name, '.obj2', 'test', 'test')

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check4():
            """Tests rsync -c works as expected."""
            self.RunGsUtil(['rsync', '-r', '-d', '-c', tmpdir, suri(bucket_uri)])
            listing1 = TailSet(tmpdir, self.FlatListDir(tmpdir))
            listing2 = TailSet(suri(bucket_uri), self.FlatListBucket(bucket_uri))
            self.assertEqual(listing1, set(['/obj1', '/.obj2', '/subdir/obj3', '/subdir/obj5', '/obj6', '/obj7']))
            self.assertEqual(listing2, set(['/obj1', '/.obj2', '/subdir/obj3', '/subdir/obj5', '/obj6', '/obj7']))
            self.assertEqual('obj6', self.RunGsUtil(['cat', suri(bucket_uri, 'obj6')], return_stdout=True))
            self._VerifyObjectMtime(bucket_uri.bucket_name, 'obj6', '100')
        _Check4()

    @SequentialAndParallelTransfer
    @unittest.skipUnless(UsingCrcmodExtension(), 'Test requires fast crcmod.')
    def test_dir_to_bucket_seek_ahead(self):
        """Tests that rsync seek-ahead iterator works correctly."""

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check1():
            """Test estimating an rsync upload operation."""
            tmpdir = self.CreateTempDir()
            subdir = os.path.join(tmpdir, 'subdir')
            os.mkdir(subdir)
            self.CreateTempFile(tmpdir=tmpdir, file_name='obj1', contents=b'obj1')
            self.CreateTempFile(tmpdir=tmpdir, file_name='.obj2', contents=b'.obj2')
            self.CreateTempFile(tmpdir=subdir, file_name='obj3', contents=b'subdir/obj3')
            bucket_uri = self.CreateBucket()
            self.CreateObject(bucket_uri=bucket_uri, object_name='.obj2', contents=b'.OBJ2')
            self.CreateObject(bucket_uri=bucket_uri, object_name='obj4', contents=b'obj4')
            self.CreateObject(bucket_uri=bucket_uri, object_name='subdir/obj5', contents=b'subdir/obj5')
            self.AssertNObjectsInBucket(bucket_uri, 3)
            with SetBotoConfigForTest([('GSUtil', 'task_estimation_threshold', '1'), ('GSUtil', 'task_estimation_force', 'True')]):
                stderr = self.RunGsUtil(['-m', 'rsync', '-d', '-r', tmpdir, suri(bucket_uri)], return_stderr=True)
                self.assertIn('Estimated work for this command: objects: 5, total size: 20', stderr)
                self.AssertNObjectsInBucket(bucket_uri, 3)
                stderr = self.RunGsUtil(['-m', 'rsync', '-d', '-r', tmpdir, suri(bucket_uri)], return_stderr=True)
                self.assertNotIn('Estimated work', stderr)
        _Check1()
        tmpdir = self.CreateTempDir(test_files=1)
        bucket_uri = self.CreateBucket()
        with SetBotoConfigForTest([('GSUtil', 'task_estimation_threshold', '0'), ('GSUtil', 'task_estimation_force', 'True')]):
            stderr = self.RunGsUtil(['-m', 'rsync', '-d', '-r', tmpdir, suri(bucket_uri)], return_stderr=True)
            self.assertNotIn('Estimated work', stderr)

    @SequentialAndParallelTransfer
    @unittest.skipUnless(UsingCrcmodExtension(), 'Test requires fast crcmod.')
    def test_dir_to_bucket_minus_d(self):
        """Tests that flat and recursive rsync dir to bucket works correctly."""
        tmpdir = self.CreateTempDir()
        subdir = os.path.join(tmpdir, 'subdir')
        os.mkdir(subdir)
        bucket_uri = self.CreateBucket()
        self.CreateTempFile(tmpdir=tmpdir, file_name='obj1', contents=b'obj1')
        o2_path = self.CreateTempFile(tmpdir=tmpdir, file_name='.obj2', contents=b'.obj2')
        self.CreateTempFile(tmpdir=subdir, file_name='obj3', contents=b'subdir/obj3')
        self.CreateObject(bucket_uri=bucket_uri, object_name='.obj2', contents=b'.OBJ2')
        self.CreateObject(bucket_uri=bucket_uri, object_name='obj4', contents=b'obj4')
        self.CreateObject(bucket_uri=bucket_uri, object_name='subdir/obj5', contents=b'subdir/obj5')
        self.AssertNObjectsInBucket(bucket_uri, 3)

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check1():
            """Tests rsync works as expected."""
            self.RunGsUtil(['rsync', '-d', tmpdir, suri(bucket_uri)])
            listing1 = TailSet(tmpdir, self.FlatListDir(tmpdir))
            listing2 = TailSet(suri(bucket_uri), self.FlatListBucket(bucket_uri))
            self.assertEqual(listing1, set(['/obj1', '/.obj2', '/subdir/obj3']))
            self.assertEqual(listing2, set(['/obj1', '/.obj2', '/subdir/obj5']))
            with open(os.path.join(tmpdir, '.obj2')) as f:
                self.assertEqual('.obj2', '\n'.join(f.readlines()))
            cloud_obj2_content = self.RunGsUtil(['cat', suri(bucket_uri, '.obj2')], return_stdout=True)
            if self._use_gcloud_storage:
                local_obj2_mtime = int(os.path.getmtime(o2_path))
                cloud_obj2_ctime = ConvertDatetimeToPOSIX(self._GetMetadataAttribute(bucket_uri.bucket_name, '.obj2', 'timeCreated'))
                self.assertTrue(cloud_obj2_content == '.obj2' or local_obj2_mtime == cloud_obj2_ctime)
            else:
                self.assertEqual('.obj2', cloud_obj2_content)
        _Check1()

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check2():
            stderr = self.RunGsUtil(['rsync', '-d', tmpdir, suri(bucket_uri)], return_stderr=True)
            self._VerifyNoChanges(stderr)
        _Check2()

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check3():
            """Tests rsync -c works as expected."""
            self.RunGsUtil(['rsync', '-d', '-c', tmpdir, suri(bucket_uri)])
            listing1 = TailSet(tmpdir, self.FlatListDir(tmpdir))
            listing2 = TailSet(suri(bucket_uri), self.FlatListBucket(bucket_uri))
            self.assertEqual(listing1, set(['/obj1', '/.obj2', '/subdir/obj3']))
            self.assertEqual(listing2, set(['/obj1', '/.obj2', '/subdir/obj5']))
            with open(os.path.join(tmpdir, '.obj2')) as f:
                self.assertEqual('.obj2', '\n'.join(f.readlines()))
            self.assertEqual('.obj2', self.RunGsUtil(['cat', suri(bucket_uri, '.obj2')], return_stdout=True))
        _Check3()

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check4():
            stderr = self.RunGsUtil(['rsync', '-d', '-c', tmpdir, suri(bucket_uri)], return_stderr=True)
            self._VerifyNoChanges(stderr)
        _Check4()
        self.CreateTempFile(tmpdir=tmpdir, file_name='obj6', contents=b'obj6')
        self.CreateObject(bucket_uri=bucket_uri, object_name='obj7', contents=b'obj7')
        os.unlink(os.path.join(tmpdir, 'obj1'))
        self.RunGsUtil(['rm', suri(bucket_uri, '.obj2')])

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check5():
            self.RunGsUtil(['rsync', '-d', '-r', tmpdir, suri(bucket_uri)])
            listing1 = TailSet(tmpdir, self.FlatListDir(tmpdir))
            listing2 = TailSet(suri(bucket_uri), self.FlatListBucket(bucket_uri))
            self.assertEqual(listing1, set(['/.obj2', '/obj6', '/subdir/obj3']))
            self.assertEqual(listing2, set(['/.obj2', '/obj6', '/subdir/obj3']))
        _Check5()

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check6():
            stderr = self.RunGsUtil(['rsync', '-d', '-r', tmpdir, suri(bucket_uri)], return_stderr=True)
            self._VerifyNoChanges(stderr)
        _Check6()

    @unittest.skipUnless(UsingCrcmodExtension(), 'Test requires fast crcmod.')
    def test_dir_to_dir_mtime(self):
        """Tests that flat and recursive rsync dir to dir works correctly."""
        tmpdir1 = self.CreateTempDir()
        tmpdir2 = self.CreateTempDir()
        subdir1 = os.path.join(tmpdir1, 'subdir1')
        subdir2 = os.path.join(tmpdir2, 'subdir2')
        os.mkdir(subdir1)
        os.mkdir(subdir2)
        self.CreateTempFile(tmpdir=tmpdir1, file_name='obj1', contents=b'obj1', mtime=10)
        self.CreateTempFile(tmpdir=tmpdir1, file_name='.obj2', contents=b'.obj2', mtime=10)
        self.CreateTempFile(tmpdir=subdir1, file_name='obj3', contents=b'subdir1/obj3', mtime=10)
        self.CreateTempFile(tmpdir=tmpdir1, file_name='obj6', contents=b'obj6', mtime=100)
        self.CreateTempFile(tmpdir=tmpdir1, file_name='obj7', contents=b'obj7_', mtime=100)
        self.CreateTempFile(tmpdir=tmpdir2, file_name='.obj2', contents=b'.OBJ2', mtime=1000)
        self.CreateTempFile(tmpdir=tmpdir2, file_name='obj4', contents=b'obj4', mtime=10)
        self.CreateTempFile(tmpdir=subdir2, file_name='obj5', contents=b'subdir2/obj5', mtime=10)
        self.CreateTempFile(tmpdir=tmpdir2, file_name='obj6', contents=b'OBJ6', mtime=100)
        self.CreateTempFile(tmpdir=tmpdir2, file_name='obj7', contents=b'obj7', mtime=100)
        self.RunGsUtil(['rsync', '-r', '-d', tmpdir1, tmpdir2])
        listing1 = TailSet(tmpdir1, self.FlatListDir(tmpdir1))
        listing2 = TailSet(tmpdir2, self.FlatListDir(tmpdir2))
        self.assertEqual(listing1, set(['/obj1', '/.obj2', '/subdir1/obj3', '/obj6', '/obj7']))
        self.assertEqual(listing2, set(['/obj1', '/.obj2', '/subdir1/obj3', '/obj6', '/obj7']))
        with open(os.path.join(tmpdir2, '.obj2')) as f:
            self.assertEqual('.obj2', '\n'.join(f.readlines()))
        with open(os.path.join(tmpdir2, 'obj6')) as f:
            self.assertEqual('OBJ6', '\n'.join(f.readlines()))
        with open(os.path.join(tmpdir2, 'obj7')) as f:
            self.assertEqual('obj7_', '\n'.join(f.readlines()))

        def _Check1():
            self.assertEqual(NO_CHANGES, self.RunGsUtil(['rsync', '-d', tmpdir1, tmpdir2], return_stderr=True))
        _Check1()
        self.RunGsUtil(['rsync', '-r', '-d', '-c', tmpdir1, tmpdir2])
        listing1 = TailSet(tmpdir1, self.FlatListDir(tmpdir1))
        listing2 = TailSet(tmpdir2, self.FlatListDir(tmpdir2))
        self.assertEqual(listing1, set(['/obj1', '/.obj2', '/subdir1/obj3', '/obj6', '/obj7']))
        self.assertEqual(listing2, set(['/obj1', '/.obj2', '/subdir1/obj3', '/obj6', '/obj7']))
        with open(os.path.join(tmpdir1, '.obj2')) as f:
            self.assertEqual('.obj2', '\n'.join(f.readlines()))
        with open(os.path.join(tmpdir1, '.obj2')) as f:
            self.assertEqual('.obj2', '\n'.join(f.readlines()))
        with open(os.path.join(tmpdir2, 'obj6')) as f:
            self.assertEqual('obj6', '\n'.join(f.readlines()))

        def _Check2():
            self.assertEqual(NO_CHANGES, self.RunGsUtil(['rsync', '-d', '-c', tmpdir1, tmpdir2], return_stderr=True))
        _Check2()
        os.unlink(os.path.join(tmpdir1, 'obj7'))
        os.unlink(os.path.join(tmpdir2, 'obj7'))
        self.CreateTempFile(tmpdir=tmpdir1, file_name='obj6', contents=b'obj6', mtime=10)
        self.CreateTempFile(tmpdir=tmpdir2, file_name='obj7', contents=b'obj7', mtime=100)
        os.unlink(os.path.join(tmpdir1, 'obj1'))
        os.unlink(os.path.join(tmpdir2, '.obj2'))
        self.RunGsUtil(['rsync', '-d', '-r', tmpdir1, tmpdir2])
        listing1 = TailSet(tmpdir1, self.FlatListDir(tmpdir1))
        listing2 = TailSet(tmpdir2, self.FlatListDir(tmpdir2))
        self.assertEqual(listing1, set(['/.obj2', '/obj6', '/subdir1/obj3']))
        self.assertEqual(listing2, set(['/.obj2', '/obj6', '/subdir1/obj3']))

        def _Check3():
            self.assertEqual(NO_CHANGES, self.RunGsUtil(['rsync', '-d', '-r', tmpdir1, tmpdir2], return_stderr=True))
        _Check3()

    @unittest.skipUnless(UsingCrcmodExtension(), 'Test requires fast crcmod.')
    def test_dir_to_dir_minus_d(self):
        """Tests that flat and recursive rsync dir to dir works correctly."""
        tmpdir1 = self.CreateTempDir()
        tmpdir2 = self.CreateTempDir()
        subdir1 = os.path.join(tmpdir1, 'subdir1')
        subdir2 = os.path.join(tmpdir2, 'subdir2')
        os.mkdir(subdir1)
        os.mkdir(subdir2)
        self.CreateTempFile(tmpdir=tmpdir1, file_name='obj1', contents=b'obj1')
        self.CreateTempFile(tmpdir=tmpdir1, file_name='.obj2', contents=b'.obj2')
        self.CreateTempFile(tmpdir=subdir1, file_name='obj3', contents=b'subdir1/obj3')
        self.CreateTempFile(tmpdir=tmpdir2, file_name='.obj2', contents=b'.OBJ2')
        self.CreateTempFile(tmpdir=tmpdir2, file_name='obj4', contents=b'obj4')
        self.CreateTempFile(tmpdir=subdir2, file_name='obj5', contents=b'subdir2/obj5')
        self.RunGsUtil(['rsync', '-d', tmpdir1, tmpdir2])
        listing1 = TailSet(tmpdir1, self.FlatListDir(tmpdir1))
        listing2 = TailSet(tmpdir2, self.FlatListDir(tmpdir2))
        self.assertEqual(listing1, set(['/obj1', '/.obj2', '/subdir1/obj3']))
        self.assertEqual(listing2, set(['/obj1', '/.obj2', '/subdir2/obj5']))
        with open(os.path.join(tmpdir1, '.obj2')) as f:
            self.assertEqual('.obj2', '\n'.join(f.readlines()))
        with open(os.path.join(tmpdir2, '.obj2')) as f:
            self.assertEqual('.OBJ2', '\n'.join(f.readlines()))

        def _Check1():
            self.assertEqual(NO_CHANGES, self.RunGsUtil(['rsync', '-d', tmpdir1, tmpdir2], return_stderr=True))
        _Check1()
        self.RunGsUtil(['rsync', '-d', '-c', tmpdir1, tmpdir2])
        listing1 = TailSet(tmpdir1, self.FlatListDir(tmpdir1))
        listing2 = TailSet(tmpdir2, self.FlatListDir(tmpdir2))
        self.assertEqual(listing1, set(['/obj1', '/.obj2', '/subdir1/obj3']))
        self.assertEqual(listing2, set(['/obj1', '/.obj2', '/subdir2/obj5']))
        with open(os.path.join(tmpdir1, '.obj2')) as f:
            self.assertEqual('.obj2', '\n'.join(f.readlines()))
        with open(os.path.join(tmpdir1, '.obj2')) as f:
            self.assertEqual('.obj2', '\n'.join(f.readlines()))

        def _Check2():
            self.assertEqual(NO_CHANGES, self.RunGsUtil(['rsync', '-d', '-c', tmpdir1, tmpdir2], return_stderr=True))
        _Check2()
        self.CreateTempFile(tmpdir=tmpdir1, file_name='obj6', contents=b'obj6')
        self.CreateTempFile(tmpdir=tmpdir2, file_name='obj7', contents=b'obj7')
        os.unlink(os.path.join(tmpdir1, 'obj1'))
        os.unlink(os.path.join(tmpdir2, '.obj2'))
        self.RunGsUtil(['rsync', '-d', '-r', tmpdir1, tmpdir2])
        listing1 = TailSet(tmpdir1, self.FlatListDir(tmpdir1))
        listing2 = TailSet(tmpdir2, self.FlatListDir(tmpdir2))
        self.assertEqual(listing1, set(['/.obj2', '/obj6', '/subdir1/obj3']))
        self.assertEqual(listing2, set(['/.obj2', '/obj6', '/subdir1/obj3']))

        def _Check3():
            self.assertEqual(NO_CHANGES, self.RunGsUtil(['rsync', '-d', '-r', tmpdir1, tmpdir2], return_stderr=True))
        _Check3()
        tmpdir1 = self.CreateTempDir()
        tmpdir2 = self.CreateTempDir()
        self.CreateTempFile(tmpdir=tmpdir1, file_name='obj1', contents=b'obj1')
        self.CreateTempFile(tmpdir=tmpdir1, file_name='obj2', contents=b'obj2')
        self.CreateTempFile(tmpdir=tmpdir2, file_name='obj2', contents=b'obj2')
        self.RunGsUtil(['rsync', '-d', tmpdir1, tmpdir2])
        listing1 = TailSet(tmpdir1, self.FlatListDir(tmpdir1))
        listing2 = TailSet(tmpdir2, self.FlatListDir(tmpdir2))
        self.assertEqual(listing1, set(['/obj1', '/obj2']))
        self.assertEqual(listing2, set(['/obj1', '/obj2']))

        def _Check4():
            self.assertEqual(NO_CHANGES, self.RunGsUtil(['rsync', '-d', tmpdir1, tmpdir2], return_stderr=True))
        _Check4()

    def test_dir_to_dir_minus_d_more_files_than_bufsize(self):
        """Tests concurrently building listing from multiple tmp file ranges."""
        tmpdir1 = self.CreateTempDir()
        tmpdir2 = self.CreateTempDir()
        for i in range(0, 1000):
            self.CreateTempFile(tmpdir=tmpdir1, file_name='d1-%s' % i, contents=b'x', mtime=i + 1)
            self.CreateTempFile(tmpdir=tmpdir2, file_name='d2-%s' % i, contents=b'y', mtime=i)
        rsync_buffer_config = [('GSUtil', 'rsync_buffer_lines', '50' if IS_WINDOWS else '2')]
        with SetBotoConfigForTest(rsync_buffer_config):
            self.RunGsUtil(['rsync', '-d', tmpdir1, tmpdir2])
        listing1 = TailSet(tmpdir1, self.FlatListDir(tmpdir1))
        listing2 = TailSet(tmpdir2, self.FlatListDir(tmpdir2))
        self.assertEqual(listing1, listing2)
        for i in range(0, 1000):
            self.assertEqual(i + 1, long(os.path.getmtime(os.path.join(tmpdir2, 'd1-%s' % i))))
            with open(os.path.join(tmpdir2, 'd1-%s' % i)) as f:
                self.assertEqual('x', '\n'.join(f.readlines()))

        def _Check():
            self.assertEqual(NO_CHANGES, self.RunGsUtil(['rsync', '-d', tmpdir1, tmpdir2], return_stderr=True))
        _Check()

    @unittest.skipUnless(UsingCrcmodExtension(), 'Test requires fast crcmod.')
    def test_bucket_to_dir_compressed_encoding(self):
        temp_file = self.CreateTempFile(contents=b'foo', file_name='bar')
        bucket_uri = self.CreateBucket()
        tmpdir = self.CreateTempDir()
        self.RunGsUtil(['cp', '-Z', temp_file, suri(bucket_uri)])
        stderr = self.RunGsUtil(['rsync', suri(bucket_uri), tmpdir], return_stderr=True)
        with open(os.path.join(tmpdir, 'bar'), 'rb') as fp:
            self.assertEqual(b'foo', fp.read())
        self.assertIn('bar has a compressed content-encoding', stderr)

    @SequentialAndParallelTransfer
    @unittest.skipUnless(UsingCrcmodExtension(), 'Test requires fast crcmod.')
    def test_bucket_to_dir_mtime(self):
        """Tests bucket to dir with mtime at the source."""
        bucket_uri = self.CreateBucket()
        tmpdir = self.CreateTempDir()
        subdir = os.path.join(tmpdir, 'subdir')
        os.mkdir(subdir)
        self.CreateObject(bucket_uri=bucket_uri, object_name='obj1', contents=b'obj1', mtime=5)
        self.CreateObject(bucket_uri=bucket_uri, object_name='.obj2', contents=b'.obj2', mtime=5)
        self.CreateObject(bucket_uri=bucket_uri, object_name='subdir/obj3', contents=b'subdir/obj3')
        self.CreateObject(bucket_uri=bucket_uri, object_name='obj4', contents=b'OBJ4')
        self.CreateObject(bucket_uri=bucket_uri, object_name='obj6', contents=b'obj6', mtime=50)
        self.CreateObject(bucket_uri=bucket_uri, object_name='obj7', contents=b'obj7', mtime=5)
        self.CreateObject(bucket_uri=bucket_uri, object_name='obj8', contents=b'obj8', mtime=100)
        self.CreateObject(bucket_uri=bucket_uri, object_name='obj9', contents=b'obj9', mtime=25)
        self.CreateObject(bucket_uri=bucket_uri, object_name='obj10', contents=b'obj10')
        time_created = ConvertDatetimeToPOSIX(self._GetMetadataAttribute(bucket_uri.bucket_name, 'obj10', 'timeCreated'))
        self.CreateObject(bucket_uri=bucket_uri, object_name='obj11', contents=b'obj11_', mtime=75)
        self.CreateTempFile(tmpdir=tmpdir, file_name='.obj2', contents=b'.OBJ2', mtime=10)
        self.CreateTempFile(tmpdir=tmpdir, file_name='obj4', contents=b'obj4', mtime=100)
        self.CreateTempFile(tmpdir=subdir, file_name='obj5', contents=b'subdir/obj5', mtime=10)
        self.CreateTempFile(tmpdir=tmpdir, file_name='obj6', contents=b'obj6', mtime=50)
        self.CreateTempFile(tmpdir=tmpdir, file_name='obj7', contents=b'OBJ7', mtime=50)
        self.CreateTempFile(tmpdir=tmpdir, file_name='obj8', contents=b'obj8', mtime=10)
        self.CreateTempFile(tmpdir=tmpdir, file_name='obj9', contents=b'OBJ9', mtime=25)
        self.CreateTempFile(tmpdir=tmpdir, file_name='obj10', contents=b'OBJ10', mtime=time_created)
        self.CreateTempFile(tmpdir=tmpdir, file_name='obj11', contents=b'obj11', mtime=75)

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check1():
            """Tests rsync works as expected."""
            self.RunGsUtil(['rsync', '-d', suri(bucket_uri), tmpdir])
            listing1 = TailSet(suri(bucket_uri), self.FlatListBucket(bucket_uri))
            listing2 = TailSet(tmpdir, self.FlatListDir(tmpdir))
            self.assertEqual(listing1, set(['/obj1', '/.obj2', '/subdir/obj3', '/obj4', '/obj6', '/obj7', '/obj8', '/obj9', '/obj10', '/obj11']))
            self.assertEqual(listing2, set(['/obj1', '/.obj2', '/obj4', '/subdir/obj5', '/obj6', '/obj7', '/obj8', '/obj9', '/obj10', '/obj11']))
            with open(os.path.join(tmpdir, '.obj2')) as f:
                self.assertEqual('.obj2', '\n'.join(f.readlines()))
            with open(os.path.join(tmpdir, 'obj4')) as f:
                self.assertEqual('OBJ4', '\n'.join(f.readlines()))
            with open(os.path.join(tmpdir, 'obj9')) as f:
                self.assertEqual('OBJ9', '\n'.join(f.readlines()))
            with open(os.path.join(tmpdir, 'obj10')) as f:
                self.assertEqual('OBJ10', '\n'.join(f.readlines()))
            with open(os.path.join(tmpdir, 'obj11')) as f:
                self.assertEqual('obj11_', '\n'.join(f.readlines()))
        _Check1()

        def _Check2():
            """Verify mtime was set for objects at destination."""
            self.assertEqual(long(os.path.getmtime(os.path.join(tmpdir, 'obj1'))), 5)
            self.assertEqual(long(os.path.getmtime(os.path.join(tmpdir, '.obj2'))), 5)
            self.assertEqual(long(os.path.getmtime(os.path.join(tmpdir, 'obj6'))), 50)
            self.assertEqual(long(os.path.getmtime(os.path.join(tmpdir, 'obj8'))), 100)
            self.assertEqual(long(os.path.getmtime(os.path.join(tmpdir, 'obj9'))), 25)
        _Check2()

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check3():
            """Tests rsync -c works as expected."""
            self.RunGsUtil(['rsync', '-r', '-d', '-c', suri(bucket_uri), tmpdir])
            listing1 = TailSet(suri(bucket_uri), self.FlatListBucket(bucket_uri))
            listing2 = TailSet(tmpdir, self.FlatListDir(tmpdir))
            self.assertEqual(listing1, set(['/obj1', '/.obj2', '/subdir/obj3', '/obj4', '/obj6', '/obj7', '/obj8', '/obj9', '/obj10', '/obj11']))
            self.assertEqual(listing2, set(['/obj1', '/.obj2', '/subdir/obj3', '/obj4', '/obj6', '/obj7', '/obj8', '/obj9', '/obj10', '/obj11']))
            self.assertEqual('obj7', self.RunGsUtil(['cat', suri(bucket_uri, 'obj7')], return_stdout=True))
            self._VerifyObjectMtime(bucket_uri.bucket_name, 'obj7', '5')
            self.assertEqual(long(os.path.getmtime(os.path.join(tmpdir, 'obj7'))), 5)
            with open(os.path.join(tmpdir, 'obj9')) as f:
                self.assertEqual('obj9', '\n'.join(f.readlines()))
            with open(os.path.join(tmpdir, 'obj10')) as f:
                self.assertEqual('obj10', '\n'.join(f.readlines()))
        _Check3()

    @unittest.skipIf(IS_WINDOWS, 'POSIX attributes not available on Windows.')
    def test_bucket_to_dir_preserve_posix_errors(self):
        """Tests that rsync -P works properly with files that would be orphaned."""
        bucket_uri = self.CreateBucket()
        tmpdir = self.CreateTempDir()
        primary_gid = os.stat(tmpdir).st_gid
        non_primary_gid = util.GetNonPrimaryGid()
        subdir = os.path.join(tmpdir, 'subdir')
        os.mkdir(subdir)
        obj1 = self.CreateObject(bucket_uri=bucket_uri, object_name='obj1', contents=b'obj1', mode='222', uid=os.getuid())
        obj2 = self.CreateObject(bucket_uri=bucket_uri, object_name='.obj2', contents=b'.obj2', gid=INVALID_GID(), mode='540')
        self.CreateObject(bucket_uri=bucket_uri, object_name='subdir/obj3', contents=b'subdir/obj3')
        obj6 = self.CreateObject(bucket_uri=bucket_uri, object_name='obj6', contents=b'obj6', gid=INVALID_GID(), mode='440')
        obj7 = self.CreateObject(bucket_uri=bucket_uri, object_name='obj7', contents=b'obj7', gid=non_primary_gid, mode='333')
        obj8 = self.CreateObject(bucket_uri=bucket_uri, object_name='obj8', contents=b'obj8', uid=INVALID_UID())
        obj9 = self.CreateObject(bucket_uri=bucket_uri, object_name='obj9', contents=b'obj9', uid=INVALID_UID(), mode='777')
        obj10 = self.CreateObject(bucket_uri=bucket_uri, object_name='obj10', contents=b'obj10', gid=INVALID_GID(), uid=INVALID_UID())
        obj11 = self.CreateObject(bucket_uri=bucket_uri, object_name='obj11', contents=b'obj11', gid=INVALID_GID(), uid=INVALID_UID(), mode='544')
        obj12 = self.CreateObject(bucket_uri=bucket_uri, object_name='obj12', contents=b'obj12', uid=INVALID_UID(), gid=USER_ID)
        obj13 = self.CreateObject(bucket_uri=bucket_uri, object_name='obj13', contents=b'obj13', uid=INVALID_UID(), gid=primary_gid, mode='644')
        obj14 = self.CreateObject(bucket_uri=bucket_uri, object_name='obj14', contents=b'obj14', uid=USER_ID, gid=INVALID_GID())
        obj15 = self.CreateObject(bucket_uri=bucket_uri, object_name='obj15', contents=b'obj15', uid=USER_ID, gid=INVALID_GID(), mode='655')
        obj16 = self.CreateObject(bucket_uri=bucket_uri, object_name='obj16', contents=b'obj16', uid=USER_ID, mode='244')
        obj17 = self.CreateObject(bucket_uri=bucket_uri, object_name='obj17', contents=b'obj17', uid=USER_ID, gid=primary_gid, mode='222')
        obj18 = self.CreateObject(bucket_uri=bucket_uri, object_name='obj18', contents=b'obj18', uid=USER_ID, gid=non_primary_gid, mode='333')
        obj19 = self.CreateObject(bucket_uri=bucket_uri, object_name='obj19', contents=b'obj19', mode='222')
        self.CreateTempFile(tmpdir=tmpdir, file_name='.obj2', contents=b'.OBJ2')
        self.CreateTempFile(tmpdir=tmpdir, file_name='obj4', contents=b'obj4')
        self.CreateTempFile(tmpdir=subdir, file_name='obj5', contents=b'subdir/obj5')

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check1():
            """Tests that an exception is thrown because files will be orphaned."""
            stderr = self.RunGsUtil(['rsync', '-P', '-r', suri(bucket_uri), tmpdir], expected_status=1, return_stderr=True)
            if self._use_gcloud_storage:
                gcloud_preserve_posix_warning = 'For preserving POSIX with rsync downloads, gsutil aborts if a single download will result in invalid destination POSIX. However, this'
                self.assertRegex(stderr, gcloud_preserve_posix_warning)
                read_regex = '{}#\\d+\\. User \\d+ owns file, but owner does not have read'
                gid_regex = "{}#\\d+ metadata doesn't exist on current system\\. GID"
                uid_regex = "{}#\\d+ metadata doesn't exist on current system\\. UID"
                self.assertRegex(stderr, read_regex.format('obj1'))
                self.assertRegex(stderr, gid_regex.format('obj2'))
                self.assertRegex(stderr, gid_regex.format('obj6'))
                self.assertRegex(stderr, read_regex.format('obj7'))
                self.assertRegex(stderr, uid_regex.format('obj8'))
                self.assertRegex(stderr, uid_regex.format('obj9'))
                self.assertRegex(stderr, uid_regex.format('obj10'))
                self.assertRegex(stderr, uid_regex.format('obj11'))
                self.assertRegex(stderr, uid_regex.format('obj12'))
                self.assertRegex(stderr, uid_regex.format('obj13'))
                self.assertRegex(stderr, gid_regex.format('obj14'))
                self.assertRegex(stderr, gid_regex.format('obj15'))
                self.assertRegex(stderr, read_regex.format('obj16'))
                self.assertRegex(stderr, read_regex.format('obj17'))
                self.assertRegex(stderr, read_regex.format('obj18'))
                self.assertRegex(stderr, read_regex.format('obj19'))
            else:
                self.assertIn(ORPHANED_FILE, stderr)
                self.assertRegex(stderr, BuildErrorRegex(obj1, POSIX_MODE_ERROR))
                self.assertRegex(stderr, BuildErrorRegex(obj2, POSIX_GID_ERROR))
                self.assertRegex(stderr, BuildErrorRegex(obj6, POSIX_GID_ERROR))
                self.assertRegex(stderr, BuildErrorRegex(obj7, POSIX_MODE_ERROR))
                self.assertRegex(stderr, BuildErrorRegex(obj8, POSIX_UID_ERROR))
                self.assertRegex(stderr, BuildErrorRegex(obj9, POSIX_UID_ERROR))
                self.assertRegex(stderr, BuildErrorRegex(obj10, POSIX_UID_ERROR))
                self.assertRegex(stderr, BuildErrorRegex(obj11, POSIX_UID_ERROR))
                self.assertRegex(stderr, BuildErrorRegex(obj12, POSIX_UID_ERROR))
                self.assertRegex(stderr, BuildErrorRegex(obj13, POSIX_UID_ERROR))
                self.assertRegex(stderr, BuildErrorRegex(obj14, POSIX_GID_ERROR))
                self.assertRegex(stderr, BuildErrorRegex(obj15, POSIX_GID_ERROR))
                self.assertRegex(stderr, BuildErrorRegex(obj16, POSIX_INSUFFICIENT_ACCESS_ERROR))
                self.assertRegex(stderr, BuildErrorRegex(obj17, POSIX_MODE_ERROR))
                self.assertRegex(stderr, BuildErrorRegex(obj18, POSIX_MODE_ERROR))
                self.assertRegex(stderr, BuildErrorRegex(obj19, POSIX_MODE_ERROR))
            listing1 = TailSet(suri(bucket_uri), self.FlatListBucket(bucket_uri))
            listing2 = TailSet(tmpdir, self.FlatListDir(tmpdir))
            self.assertEqual(listing1, set(['/obj1', '/.obj2', '/subdir/obj3', '/obj6', '/obj7', '/obj8', '/obj9', '/obj10', '/obj11', '/obj12', '/obj13', '/obj14', '/obj15', '/obj16', '/obj17', '/obj18', '/obj19']))
            if self._use_gcloud_storage:
                self.assertEqual(listing2, set(['/.obj2', '/obj4', '/subdir/obj3', '/subdir/obj5']))
            else:
                self.assertEqual(listing2, set(['/.obj2', '/obj4', '/subdir/obj5']))
        _Check1()
        self._SetObjectCustomMetadataAttribute(self.default_provider, bucket_uri.bucket_name, '.obj2', MODE_ATTR, '640')

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check2():
            """Tests that a file with a valid mode in metadata, nothing changed."""
            stderr = self.RunGsUtil(['rsync', '-P', '-r', suri(bucket_uri), tmpdir], expected_status=1, return_stderr=True)
            if self._use_gcloud_storage:
                self.assertIn("doesn't exist on current system. GID:", stderr)
            else:
                self.assertIn(ORPHANED_FILE, stderr)
            listing1 = TailSet(suri(bucket_uri), self.FlatListBucket(bucket_uri))
            listing2 = TailSet(tmpdir, self.FlatListDir(tmpdir))
            self.assertEqual(listing1, set(['/obj1', '/.obj2', '/subdir/obj3', '/obj6', '/obj7', '/obj8', '/obj9', '/obj10', '/obj11', '/obj12', '/obj13', '/obj14', '/obj15', '/obj16', '/obj17', '/obj18', '/obj19']))
            if self._use_gcloud_storage:
                self.assertEqual(listing2, set(['/.obj2', '/obj4', '/subdir/obj3', '/subdir/obj5']))
            else:
                self.assertEqual(listing2, set(['/.obj2', '/obj4', '/subdir/obj5']))
        _Check2()

    @SequentialAndParallelTransfer
    @unittest.skipIf(IS_WINDOWS, 'POSIX attributes not available on Windows.')
    @unittest.skipUnless(UsingCrcmodExtension(), 'Test requires fast crcmod.')
    def test_bucket_to_dir_preserve_posix_no_errors(self):
        """Tests that rsync -P works properly with default file attributes."""
        bucket_uri = self.CreateBucket()
        tmpdir = self.CreateTempDir()
        primary_gid = os.stat(tmpdir).st_gid
        non_primary_gid = util.GetNonPrimaryGid()
        subdir = os.path.join(tmpdir, 'subdir')
        os.mkdir(subdir)
        self.CreateObject(bucket_uri=bucket_uri, object_name='obj1', contents=b'obj1', mode='444')
        self.CreateObject(bucket_uri=bucket_uri, object_name='.obj2', contents=b'.obj2', gid=primary_gid)
        self.CreateObject(bucket_uri=bucket_uri, object_name='subdir/obj3', contents=b'subdir/obj3', gid=non_primary_gid)
        self.CreateObject(bucket_uri=bucket_uri, object_name='obj6', contents=b'obj6', gid=primary_gid, mode='555')
        self.CreateObject(bucket_uri=bucket_uri, object_name='obj7', contents=b'obj7', gid=non_primary_gid, mode='444')
        self.CreateObject(bucket_uri=bucket_uri, object_name='obj8', contents=b'obj8', uid=USER_ID)
        self.CreateObject(bucket_uri=bucket_uri, object_name='obj9', contents=b'obj9', uid=USER_ID, mode='422')
        self.CreateObject(bucket_uri=bucket_uri, object_name='obj10', contents=b'obj10', uid=USER_ID, gid=primary_gid)
        self.CreateObject(bucket_uri=bucket_uri, object_name='obj11', contents=b'obj11', uid=USER_ID, gid=non_primary_gid)
        self.CreateObject(bucket_uri=bucket_uri, object_name='obj12', contents=b'obj12', uid=USER_ID, gid=primary_gid, mode='400')
        self.CreateObject(bucket_uri=bucket_uri, object_name='obj13', contents=b'obj13', uid=USER_ID, gid=non_primary_gid, mode='533')
        self.CreateObject(bucket_uri=bucket_uri, object_name='obj14', contents=b'obj14', uid=USER_ID, mode='444')
        self.CreateTempFile(tmpdir=tmpdir, file_name='.obj2', contents=b'.OBJ2')
        self.CreateTempFile(tmpdir=tmpdir, file_name='obj4', contents=b'obj4')
        self.CreateTempFile(tmpdir=subdir, file_name='obj5', contents=b'subdir/obj5')

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check1():
            """Verifies that all attributes were copied correctly when -P is used."""
            self.RunGsUtil(['rsync', '-P', '-r', suri(bucket_uri), tmpdir])
            listing1 = TailSet(suri(bucket_uri), self.FlatListBucket(bucket_uri))
            listing2 = TailSet(tmpdir, self.FlatListDir(tmpdir))
            self.assertEqual(listing1, set(['/obj1', '/.obj2', '/subdir/obj3', '/obj6', '/obj7', '/obj8', '/obj9', '/obj10', '/obj11', '/obj12', '/obj13', '/obj14']))
            self.assertEqual(listing2, set(['/obj1', '/.obj2', '/subdir/obj3', '/obj4', '/subdir/obj5', '/obj6', '/obj7', '/obj8', '/obj9', '/obj10', '/obj11', '/obj12', '/obj13', '/obj14']))
        _Check1()
        self.VerifyLocalPOSIXPermissions(os.path.join(tmpdir, 'obj1'), uid=os.getuid(), mode=292)
        self.VerifyLocalPOSIXPermissions(os.path.join(tmpdir, '.obj2'), gid=primary_gid, uid=os.getuid(), mode=DEFAULT_MODE)
        self.VerifyLocalPOSIXPermissions(os.path.join(subdir, 'obj3'), gid=non_primary_gid, mode=DEFAULT_MODE)
        self.VerifyLocalPOSIXPermissions(os.path.join(tmpdir, 'obj6'), gid=primary_gid, mode=365)
        self.VerifyLocalPOSIXPermissions(os.path.join(tmpdir, 'obj7'), gid=non_primary_gid, mode=292)
        self.VerifyLocalPOSIXPermissions(os.path.join(tmpdir, 'obj8'), gid=primary_gid, mode=DEFAULT_MODE)
        self.VerifyLocalPOSIXPermissions(os.path.join(tmpdir, 'obj9'), uid=USER_ID, mode=274)
        self.VerifyLocalPOSIXPermissions(os.path.join(tmpdir, 'obj10'), uid=USER_ID, gid=primary_gid, mode=DEFAULT_MODE)
        self.VerifyLocalPOSIXPermissions(os.path.join(tmpdir, 'obj11'), uid=USER_ID, gid=non_primary_gid, mode=DEFAULT_MODE)
        self.VerifyLocalPOSIXPermissions(os.path.join(tmpdir, 'obj12'), uid=USER_ID, gid=primary_gid, mode=256)
        self.VerifyLocalPOSIXPermissions(os.path.join(tmpdir, 'obj13'), uid=USER_ID, gid=non_primary_gid, mode=347)
        self.VerifyLocalPOSIXPermissions(os.path.join(tmpdir, 'obj14'), uid=USER_ID, mode=292)

    @unittest.skipUnless(UsingCrcmodExtension(), 'Test requires fast crcmod.')
    def test_bucket_to_dir_minus_d(self):
        """Tests that flat and recursive rsync bucket to dir works correctly."""
        bucket_uri = self.CreateBucket()
        tmpdir = self.CreateTempDir()
        subdir = os.path.join(tmpdir, 'subdir')
        os.mkdir(subdir)
        self.CreateObject(bucket_uri=bucket_uri, object_name='obj1', contents=b'obj1')
        self.CreateObject(bucket_uri=bucket_uri, object_name='.obj2', contents=b'.obj2', mtime=0)
        self.CreateObject(bucket_uri=bucket_uri, object_name='subdir/obj3', contents=b'subdir/obj3')
        self.CreateTempFile(tmpdir=tmpdir, file_name='.obj2', contents=b'.OBJ2')
        self.CreateTempFile(tmpdir=tmpdir, file_name='obj4', contents=b'obj4')
        self.CreateTempFile(tmpdir=subdir, file_name='obj5', contents=b'subdir/obj5')

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check1():
            """Tests rsync works as expected."""
            self.RunGsUtil(['rsync', '-d', suri(bucket_uri), tmpdir])
            listing1 = TailSet(suri(bucket_uri), self.FlatListBucket(bucket_uri))
            listing2 = TailSet(tmpdir, self.FlatListDir(tmpdir))
            self.assertEqual(listing1, set(['/obj1', '/.obj2', '/subdir/obj3']))
            self.assertEqual(listing2, set(['/obj1', '/.obj2', '/subdir/obj5']))
            self.assertEqual('.obj2', self.RunGsUtil(['cat', suri(bucket_uri, '.obj2')], return_stdout=True))
            with open(os.path.join(tmpdir, '.obj2')) as f:
                self.assertEqual('.obj2', '\n'.join(f.readlines()))
        _Check1()

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check2():
            stderr = self.RunGsUtil(['rsync', '-d', suri(bucket_uri), tmpdir], return_stderr=True)
            self._VerifyNoChanges(stderr)
        _Check2()

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check3():
            """Tests rsync -c works as expected."""
            self.RunGsUtil(['rsync', '-d', '-c', suri(bucket_uri), tmpdir])
            listing1 = TailSet(suri(bucket_uri), self.FlatListBucket(bucket_uri))
            listing2 = TailSet(tmpdir, self.FlatListDir(tmpdir))
            self.assertEqual(listing1, set(['/obj1', '/.obj2', '/subdir/obj3']))
            self.assertEqual(listing2, set(['/obj1', '/.obj2', '/subdir/obj5']))
            self.assertEqual('.obj2', self.RunGsUtil(['cat', suri(bucket_uri, '.obj2')], return_stdout=True))
            with open(os.path.join(tmpdir, '.obj2')) as f:
                self.assertEqual('.obj2', '\n'.join(f.readlines()))
        _Check3()

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check4():
            stderr = self.RunGsUtil(['rsync', '-d', '-c', suri(bucket_uri), tmpdir], return_stderr=True)
            self._VerifyNoChanges(stderr)
        _Check4()
        self.CreateObject(bucket_uri=bucket_uri, object_name='obj6', contents=b'obj6')
        self.CreateTempFile(tmpdir=tmpdir, file_name='obj7', contents=b'obj7')
        self.RunGsUtil(['rm', suri(bucket_uri, 'obj1')])
        os.unlink(os.path.join(tmpdir, '.obj2'))

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check5():
            self.RunGsUtil(['rsync', '-d', '-r', suri(bucket_uri), tmpdir])
            listing1 = TailSet(suri(bucket_uri), self.FlatListBucket(bucket_uri))
            listing2 = TailSet(tmpdir, self.FlatListDir(tmpdir))
            self.assertEqual(listing1, set(['/.obj2', '/obj6', '/subdir/obj3']))
            self.assertEqual(listing2, set(['/.obj2', '/obj6', '/subdir/obj3']))
        _Check5()

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check6():
            stderr = self.RunGsUtil(['rsync', '-d', '-r', suri(bucket_uri), tmpdir], return_stderr=True)
            self._VerifyNoChanges(stderr)
        _Check6()

    @unittest.skipUnless(UsingCrcmodExtension(), 'Test requires fast crcmod.')
    def test_bucket_to_dir_minus_d_with_fname_case_change(self):
        """Tests that name case changes work correctly.

    Example:

    Windows filenames are case-preserving in what you wrote, but case-
    insensitive when compared. If you synchronize from FS to cloud and then
    change case-naming in local files, you could end up with this situation:

    Cloud copy is called .../TiVo/...
    FS copy is called      .../Tivo/...

    Then, if you rsync from cloud to FS, if rsync doesn't recognize that on
    Windows these names are identical, each rsync run will cause both a copy
    and a delete to be executed.
    """
        bucket_uri = self.CreateBucket()
        tmpdir = self.CreateTempDir()
        self.CreateObject(bucket_uri=bucket_uri, object_name='obj1', contents=b'obj1')
        self.CreateTempFile(tmpdir=tmpdir, file_name='Obj1', contents=b'obj1')

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check1():
            """Tests rsync works as expected."""
            output = self.RunGsUtil(['rsync', '-d', '-r', suri(bucket_uri), tmpdir], return_stderr=True)
            if IS_WINDOWS:
                self.assertEqual(NO_CHANGES, output)
            else:
                self.assertNotEqual(NO_CHANGES, output)
        _Check1()

    @unittest.skipUnless(UsingCrcmodExtension(), 'Test requires fast crcmod.')
    @SkipForS3('The boto lib used for S3 does not handle objects starting with slashes if we use V4 signature.')
    def test_bucket_to_dir_minus_d_with_leftover_dir_placeholder(self):
        """Tests that we correctly handle leftover dir placeholders.

    See comments in gslib.commands.rsync._FieldedListingIterator for details.
    """
        bucket_uri = self.CreateBucket()
        tmpdir = self.CreateTempDir()
        self.CreateObject(bucket_uri=bucket_uri, object_name='obj1', contents=b'obj1')
        key_uri = self.StorageUriCloneReplaceName(bucket_uri, '/')
        self.StorageUriSetContentsFromString(key_uri, '')

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check1():
            """Tests rsync works as expected."""
            self.RunGsUtil(['rsync', '-d', '-r', suri(bucket_uri), tmpdir], return_stderr=True)
            listing1 = TailSet(suri(bucket_uri), self.FlatListBucket(bucket_uri))
            listing2 = TailSet(tmpdir, self.FlatListDir(tmpdir))
            self.assertEqual(listing1, set(['/obj1', '//']))
            self.assertEqual(listing2, set(['/obj1']))
        _Check1()

    @unittest.skipIf(IS_WINDOWS, 'os.symlink() is not available on Windows.')
    def test_rsync_minus_r_minus_e(self):
        """Tests that rsync -e -r ignores symlinks when recursing."""
        bucket_uri = self.CreateBucket()
        tmpdir = self.CreateTempDir()
        subdir = os.path.join(tmpdir, 'subdir')
        os.mkdir(subdir)
        os.mkdir(os.path.join(tmpdir, 'missing'))
        os.symlink(os.path.join(tmpdir, 'missing'), os.path.join(subdir, 'missing'))
        os.rmdir(os.path.join(tmpdir, 'missing'))
        self.RunGsUtil(['rsync', '-r', '-e', tmpdir, suri(bucket_uri)])

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

    @SkipForS3('S3 does not support composite objects')
    def test_bucket_to_bucket_minus_d_with_composites(self):
        """Tests that rsync works with composite objects (which don't have MD5s)."""
        bucket1_uri = self.CreateBucket()
        bucket2_uri = self.CreateBucket()
        self.CreateObject(bucket_uri=bucket1_uri, object_name='obj1', contents=b'obj1')
        self.CreateObject(bucket_uri=bucket1_uri, object_name='.obj2', contents=b'.obj2')
        self.RunGsUtil(['compose', suri(bucket1_uri, 'obj1'), suri(bucket1_uri, '.obj2'), suri(bucket1_uri, 'obj3')])
        self.CreateObject(bucket_uri=bucket2_uri, object_name='.obj2', contents=b'.OBJ2')
        self.CreateObject(bucket_uri=bucket2_uri, object_name='obj4', contents=b'obj4')

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check1():
            self.RunGsUtil(['rsync', '-d', suri(bucket1_uri), suri(bucket2_uri)])
            listing1 = TailSet(suri(bucket1_uri), self.FlatListBucket(bucket1_uri))
            listing2 = TailSet(suri(bucket2_uri), self.FlatListBucket(bucket2_uri))
            self.assertEqual(listing1, set(['/obj1', '/.obj2', '/obj3']))
            self.assertEqual(listing2, set(['/obj1', '/.obj2', '/obj3']))
        _Check1()

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check2():
            stderr = self.RunGsUtil(['rsync', '-d', suri(bucket1_uri), suri(bucket2_uri)], return_stderr=True)
            self._VerifyNoChanges(stderr)
        _Check2()

    def test_bucket_to_bucket_minus_d_empty_dest(self):
        """Tests working with empty dest bucket (iter runs out before src iter)."""
        bucket1_uri = self.CreateBucket()
        bucket2_uri = self.CreateBucket()
        self.CreateObject(bucket_uri=bucket1_uri, object_name='obj1', contents=b'obj1')
        self.CreateObject(bucket_uri=bucket1_uri, object_name='.obj2', contents=b'.obj2')

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check1():
            self.RunGsUtil(['rsync', '-d', suri(bucket1_uri), suri(bucket2_uri)])
            listing1 = TailSet(suri(bucket1_uri), self.FlatListBucket(bucket1_uri))
            listing2 = TailSet(suri(bucket2_uri), self.FlatListBucket(bucket2_uri))
            self.assertEqual(listing1, set(['/obj1', '/.obj2']))
            self.assertEqual(listing2, set(['/obj1', '/.obj2']))
        _Check1()

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check2():
            stderr = self.RunGsUtil(['rsync', '-d', suri(bucket1_uri), suri(bucket2_uri)], return_stderr=True)
            self._VerifyNoChanges(stderr)
        _Check2()

    def test_bucket_to_bucket_minus_d_empty_src(self):
        """Tests working with empty src bucket (iter runs out before dst iter)."""
        bucket1_uri = self.CreateBucket()
        bucket2_uri = self.CreateBucket()
        self.CreateObject(bucket_uri=bucket2_uri, object_name='obj1', contents=b'obj1')
        self.CreateObject(bucket_uri=bucket2_uri, object_name='.obj2', contents=b'.obj2')

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check1():
            self.RunGsUtil(['rsync', '-d', suri(bucket1_uri), suri(bucket2_uri)])
            stderr = self.RunGsUtil(['ls', suri(bucket1_uri, '**')], expected_status=1, return_stderr=True)
            self.assertIn('One or more URLs matched no objects', stderr)
            stderr = self.RunGsUtil(['ls', suri(bucket2_uri, '**')], expected_status=1, return_stderr=True)
            self.assertIn('One or more URLs matched no objects', stderr)
        _Check1()

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check2():
            stderr = self.RunGsUtil(['rsync', '-d', suri(bucket1_uri), suri(bucket2_uri)], return_stderr=True)
            self._VerifyNoChanges(stderr)
        _Check2()

    def test_rsync_minus_d_minus_p(self):
        """Tests that rsync -p preserves ACLs."""
        bucket1_uri = self.CreateBucket()
        bucket2_uri = self.CreateBucket()
        self.CreateObject(bucket_uri=bucket1_uri, object_name='obj1', contents=b'obj1')
        self.RunGsUtil(['acl', 'set', 'public-read', suri(bucket1_uri, 'obj1')])

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check1():
            """Tests rsync -p works as expected."""
            self.RunGsUtil(['rsync', '-d', '-p', suri(bucket1_uri), suri(bucket2_uri)])
            listing1 = TailSet(suri(bucket1_uri), self.FlatListBucket(bucket1_uri))
            listing2 = TailSet(suri(bucket2_uri), self.FlatListBucket(bucket2_uri))
            self.assertEqual(listing1, set(['/obj1']))
            self.assertEqual(listing2, set(['/obj1']))
            acl1_json = self.RunGsUtil(['acl', 'get', suri(bucket1_uri, 'obj1')], return_stdout=True)
            acl2_json = self.RunGsUtil(['acl', 'get', suri(bucket2_uri, 'obj1')], return_stdout=True)
            self.assertEqual(acl1_json, acl2_json)
        _Check1()

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check2():
            stderr = self.RunGsUtil(['rsync', '-d', '-p', suri(bucket1_uri), suri(bucket2_uri)], return_stderr=True)
            self._VerifyNoChanges(stderr)
        _Check2()

    def test_rsync_canned_acl(self):
        """Tests that rsync -a applies ACLs."""
        bucket1_uri = self.CreateBucket()
        bucket2_uri = self.CreateBucket()
        self.CreateObject(bucket_uri=bucket1_uri, object_name='obj1', contents=b'obj1')
        self.RunGsUtil(['acl', 'get', suri(bucket1_uri, 'obj1')])

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check():
            """Tests rsync -a works as expected."""
            self.RunGsUtil(['rsync', '-d', '-a', 'public-read', suri(bucket1_uri), suri(bucket2_uri)])
            listing1 = TailSet(suri(bucket1_uri), self.FlatListBucket(bucket1_uri))
            listing2 = TailSet(suri(bucket2_uri), self.FlatListBucket(bucket2_uri))
            self.assertEqual(listing1, set(['/obj1']))
            self.assertEqual(listing2, set(['/obj1']))
            self.RunGsUtil(['acl', 'set', 'public-read', suri(bucket1_uri, 'obj1')])
            acl1_json = self.RunGsUtil(['acl', 'get', suri(bucket1_uri, 'obj1')], return_stdout=True)
            acl2_json = self.RunGsUtil(['acl', 'get', suri(bucket2_uri, 'obj1')], return_stdout=True)
            self.assertEqual(acl1_json, acl2_json)
        _Check()

    def test_rsync_to_nonexistent_bucket_subdir(self):
        """Tests that rsync to non-existent bucket subdir works."""
        tmpdir = self.CreateTempDir()
        subdir = os.path.join(tmpdir, 'subdir')
        os.mkdir(subdir)
        bucket_url = self.CreateBucket()
        self.CreateTempFile(tmpdir=tmpdir, file_name='obj1', contents=b'obj1')
        self.CreateTempFile(tmpdir=tmpdir, file_name='.obj2', contents=b'.obj2')
        self.CreateTempFile(tmpdir=subdir, file_name='obj3', contents=b'subdir/obj3')

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check1():
            """Tests rsync works as expected."""
            self.RunGsUtil(['rsync', '-r', tmpdir, suri(bucket_url, 'subdir')])
            listing1 = TailSet(tmpdir, self.FlatListDir(tmpdir))
            listing2 = TailSet(suri(bucket_url, 'subdir'), self.FlatListBucket(self.StorageUriCloneReplaceName(bucket_url, 'subdir')))
            self.assertEqual(listing1, set(['/obj1', '/.obj2', '/subdir/obj3']))
            self.assertEqual(listing2, set(['/obj1', '/.obj2', '/subdir/obj3']))
        _Check1()

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check2():
            stderr = self.RunGsUtil(['rsync', '-r', tmpdir, suri(bucket_url, 'subdir')], return_stderr=True)
            self._VerifyNoChanges(stderr)
        _Check2()

    def test_rsync_to_nonexistent_bucket_subdir_prefix_of_existing_obj(self):
        """Tests that rsync with destination url as a prefix of existing obj works.

    Test to make sure that a dir/subdir gets created if it does not exist
    even when the new dir is a prefix of an existing dir.
    e.g if gs://some_bucket/foobar exists, and we run the command
    rsync some_dir gs://some_bucket/foo
    this should create a subdir foo
    """
        tmpdir = self.CreateTempDir()
        bucket_url = self.CreateBucket()
        self.CreateObject(bucket_uri=bucket_url, object_name='foobar', contents=b'obj1')
        self.CreateTempFile(tmpdir=tmpdir, file_name='obj1', contents=b'obj1')
        self.CreateTempFile(tmpdir=tmpdir, file_name='.obj2', contents=b'.obj2')

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check1():
            """Tests rsync works as expected."""
            self.RunGsUtil(['rsync', '-r', tmpdir, suri(bucket_url, 'foo')])
            listing1 = TailSet(tmpdir, self.FlatListDir(tmpdir))
            listing2 = TailSet(suri(bucket_url, 'foo'), self.FlatListBucket(self.StorageUriCloneReplaceName(bucket_url, 'foo')))
            self.assertEqual(listing1, set(['/obj1', '/.obj2']))
            self.assertEqual(listing2, set(['/obj1', '/.obj2']))
        _Check1()

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check2():
            stderr = self.RunGsUtil(['rsync', '-r', tmpdir, suri(bucket_url, 'foo')], return_stderr=True)
            self._VerifyNoChanges(stderr)
        _Check2()

    def test_rsync_from_nonexistent_bucket(self):
        """Tests that rsync from a non-existent bucket subdir fails gracefully."""
        tmpdir = self.CreateTempDir()
        self.CreateTempFile(tmpdir=tmpdir, file_name='obj1', contents=b'obj1')
        self.CreateTempFile(tmpdir=tmpdir, file_name='.obj2', contents=b'.obj2')
        bucket_url_str = '%s://%s' % (self.default_provider, self.nonexistent_bucket_name)

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check():
            stderr = self.RunGsUtil(['rsync', '-d', bucket_url_str, tmpdir], expected_status=1, return_stderr=True)
            if self._use_gcloud_storage:
                self.assertIn('not found: 404', stderr)
            else:
                self.assertIn('Caught non-retryable exception', stderr)
            listing = TailSet(tmpdir, self.FlatListDir(tmpdir))
            self.assertEqual(listing, set(['/obj1', '/.obj2']))
        _Check()

    def test_rsync_to_nonexistent_bucket(self):
        """Tests that rsync from a non-existent bucket subdir fails gracefully."""
        tmpdir = self.CreateTempDir()
        self.CreateTempFile(tmpdir=tmpdir, file_name='obj1', contents=b'obj1')
        self.CreateTempFile(tmpdir=tmpdir, file_name='.obj2', contents=b'.obj2')
        bucket_url_str = '%s://%s' % (self.default_provider, self.nonexistent_bucket_name)

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check():
            stderr = self.RunGsUtil(['rsync', '-d', bucket_url_str, tmpdir], expected_status=1, return_stderr=True)
            if self._use_gcloud_storage:
                self.assertIn('not found: 404', stderr)
            else:
                self.assertIn('Caught non-retryable exception', stderr)
            listing = TailSet(tmpdir, self.FlatListDir(tmpdir))
            self.assertEqual(listing, set(['/obj1', '/.obj2']))
        _Check()

    def test_bucket_to_bucket_minus_d_with_overwrite_and_punc_chars(self):
        """Tests that punc chars in filenames don't confuse sort order."""
        bucket1_uri = self.CreateBucket()
        bucket2_uri = self.CreateBucket()
        self.CreateObject(bucket_uri=bucket1_uri, object_name='e/obj1', contents=b'obj1')
        self.CreateObject(bucket_uri=bucket1_uri, object_name='e-1/.obj2', contents=b'.obj2')
        self.CreateObject(bucket_uri=bucket2_uri, object_name='e/obj1', contents=b'OBJ1')
        self.CreateObject(bucket_uri=bucket2_uri, object_name='e-1/.obj2', contents=b'.obj2')
        self.AssertNObjectsInBucket(bucket1_uri, 2)
        self.AssertNObjectsInBucket(bucket2_uri, 2)

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check1():
            """Tests rsync works as expected."""
            self.RunGsUtil(['rsync', '-rd', suri(bucket1_uri), suri(bucket2_uri)])
            listing1 = TailSet(suri(bucket1_uri), self.FlatListBucket(bucket1_uri))
            listing2 = TailSet(suri(bucket2_uri), self.FlatListBucket(bucket2_uri))
            self.assertEqual(listing1, set(['/e/obj1', '/e-1/.obj2']))
            self.assertEqual(listing2, set(['/e/obj1', '/e-1/.obj2']))
            self.assertEqual('obj1', self.RunGsUtil(['cat', suri(bucket2_uri, 'e/obj1')], return_stdout=True))
            self.assertEqual('.obj2', self.RunGsUtil(['cat', suri(bucket2_uri, 'e-1/.obj2')], return_stdout=True))
        _Check1()

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check2():
            stderr = self.RunGsUtil(['rsync', '-d', suri(bucket1_uri), suri(bucket2_uri)], return_stderr=True)
            self._VerifyNoChanges(stderr)
        _Check2()

    def _test_dir_to_bucket_regex_paramaterized(self, flag):
        """Tests that rsync regex exclusions work correctly."""
        tmpdir = self.CreateTempDir()
        bucket_uri = self.CreateBucket()
        self.CreateTempFile(tmpdir=tmpdir, file_name='obj1', contents=b'obj1')
        self.CreateTempFile(tmpdir=tmpdir, file_name='.obj2', contents=b'.obj2')
        self.CreateTempFile(tmpdir=tmpdir, file_name='obj3', contents=b'obj3')
        self.CreateObject(bucket_uri=bucket_uri, object_name='.obj2', contents=b'.obj2')
        self.CreateObject(bucket_uri=bucket_uri, object_name='obj4', contents=b'obj4')
        self.CreateObject(bucket_uri=bucket_uri, object_name='obj5', contents=b'obj5')
        self.AssertNObjectsInBucket(bucket_uri, 3)

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check1():
            """Tests rsync works as expected."""
            self.RunGsUtil(['rsync', '-d', flag, 'obj[34]', tmpdir, suri(bucket_uri)])
            listing1 = TailSet(tmpdir, self.FlatListDir(tmpdir))
            listing2 = TailSet(suri(bucket_uri), self.FlatListBucket(bucket_uri))
            self.assertEqual(listing1, set(['/obj1', '/.obj2', '/obj3']))
            self.assertEqual(listing2, set(['/obj1', '/.obj2', '/obj4']))
        _Check1()

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check2():
            stderr = self.RunGsUtil(['rsync', '-d', flag, 'obj[34]', tmpdir, suri(bucket_uri)], return_stderr=True)
            self._VerifyNoChanges(stderr)
        _Check2()

    def test_dir_to_bucket_minus_x(self):
        """Tests that rsync regex exclusions work correctly for -x."""
        self._test_dir_to_bucket_regex_paramaterized('-x')

    def test_dir_to_bucket_minus_y(self):
        """Tests that rsync regex exclusions work correctly for -y."""
        self._test_dir_to_bucket_regex_paramaterized('-y')

    def _test_dir_to_bucket_regex_negative_lookahead(self, flag, includes):
        """Tests if negative lookahead includes files/objects."""
        bucket_uri = self.CreateBucket()
        tmpdir = self.CreateTempDir(test_files=['a', 'b', 'c', ('data1', 'a.txt'), ('data1', 'ok'), ('data2', 'b.txt'), ('data3', 'data4', 'c.txt')])
        self.RunGsUtil(['rsync', '-r', flag, '^(?!.*\\.txt$).*', tmpdir, suri(bucket_uri)], return_stderr=True)
        listing = TailSet(tmpdir, self.FlatListDir(tmpdir))
        self.assertEqual(listing, set(['/a', '/b', '/c', '/data1/a.txt', '/data1/ok', '/data2/b.txt', '/data3/data4/c.txt']))
        if includes:
            actual = TailSet(suri(bucket_uri), self.FlatListBucket(bucket_uri))
            self.assertEqual(actual, set(['/data1/a.txt', '/data2/b.txt', '/data3/data4/c.txt']))
        else:
            stderr = self.RunGsUtil(['ls', suri(bucket_uri, '**')], expected_status=1, return_stderr=True)
            self.assertIn('One or more URLs matched no objects', stderr)

    def test_dir_to_bucket_negative_lookahead_works_in_minus_x(self):
        """Test that rsync -x negative lookahead includes objects/files."""
        self._test_dir_to_bucket_regex_negative_lookahead('-x', includes=True)

    def test_dir_to_bucket_negative_lookahead_breaks_in_minux_y(self):
        """Test that rsync -y nevative lookahead does not includes objects/files."""
        self._test_dir_to_bucket_regex_negative_lookahead('-y', includes=False)

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

    def test_dir_to_bucket_relative_minus_x(self):
        """Test that rsync -x option works with a relative regex per the docs."""
        self._test_dir_to_bucket_relative_regex_paramaterized('-x', skip_dirs=False)

    def test_dir_to_bucket_relative_minus_y(self):
        """Test that rsync -y option works with a relative regex per the docs."""
        self._test_dir_to_bucket_relative_regex_paramaterized('-y', skip_dirs=True)

    @unittest.skipIf(IS_WINDOWS, "os.chmod() won't make file unreadable on Windows.")
    def test_dir_to_bucket_minus_C(self):
        """Tests that rsync -C option works correctly."""
        tmpdir = self.CreateTempDir()
        bucket_uri = self.CreateBucket()
        self.CreateTempFile(tmpdir=tmpdir, file_name='obj1', contents=b'obj1')
        path = self.CreateTempFile(tmpdir=tmpdir, file_name='obj2', contents=b'obj2')
        os.chmod(path, 0)
        self.CreateTempFile(tmpdir=tmpdir, file_name='obj3', contents=b'obj3')

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check():
            """Tests rsync works as expected."""
            stderr = self.RunGsUtil(['rsync', '-C', tmpdir, suri(bucket_uri)], expected_status=1, return_stderr=True)
            if self._use_gcloud_storage:
                self.assertIn("Permission denied: '{}'".format(path), stderr)
            else:
                self.assertIn('1 files/objects could not be copied/removed.', stderr)
            listing1 = TailSet(tmpdir, self.FlatListDir(tmpdir))
            listing2 = TailSet(suri(bucket_uri), self.FlatListBucket(bucket_uri))
            self.assertEqual(listing1, set(['/obj1', '/obj2', '/obj3']))
            self.assertEqual(listing2, set(['/obj1', '/obj3']))
        _Check()

    @unittest.skipIf(IS_WINDOWS, 'Windows Unicode support is problematic in Python 2.x.')
    def test_dir_to_bucket_with_unicode_chars(self):
        """Tests that rsync -r works correctly with unicode filenames."""
        tmpdir = self.CreateTempDir()
        bucket_uri = self.CreateBucket()
        file_list = ['morales_suenÌƒos.jpg', 'morales_suenÌƒos.jpg', 'fooꝾoo']
        for filename in file_list:
            self.CreateTempFile(tmpdir=tmpdir, file_name=filename)
        expected_list_results = frozenset(['/morales_suenÌƒos.jpg', '/fooꝾoo']) if IS_OSX else frozenset(['/morales_suenÌƒos.jpg', '/morales_suenÌƒos.jpg', '/fooꝾoo'])

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check():
            """Tests rsync works as expected."""
            self.RunGsUtil(['rsync', '-r', tmpdir, suri(bucket_uri)])
            listing1 = TailSet(tmpdir, self.FlatListDir(tmpdir))
            listing2 = TailSet(suri(bucket_uri), self.FlatListBucket(bucket_uri))
            self.assertEqual(set(listing1), expected_list_results)
            self.assertEqual(set(listing2), expected_list_results)
        _Check()

    def test_dir_to_bucket_minus_u(self):
        """Tests that rsync -u works correctly."""
        tmpdir = self.CreateTempDir()
        dst_bucket = self.CreateBucket()
        ORIG_MTIME = 10
        self.CreateObject(bucket_uri=dst_bucket, object_name='obj1', contents=b'obj1-1', mtime=ORIG_MTIME)
        self.CreateObject(bucket_uri=dst_bucket, object_name='obj2', contents=b'obj2-1', mtime=ORIG_MTIME)
        self.CreateObject(bucket_uri=dst_bucket, object_name='obj3', contents=b'obj3-1', mtime=ORIG_MTIME)
        self.CreateObject(bucket_uri=dst_bucket, object_name='obj4', contents=b'obj4-1', mtime=ORIG_MTIME)
        self.CreateObject(bucket_uri=dst_bucket, object_name='obj5', contents=b'obj5-1', mtime=ORIG_MTIME)
        self.CreateObject(bucket_uri=dst_bucket, object_name='obj6', contents=b'obj6-1', mtime=ORIG_MTIME)
        self.CreateTempFile(tmpdir=tmpdir, file_name='obj1', contents=b'obj1-2', mtime=ORIG_MTIME - 1)
        self.CreateTempFile(tmpdir=tmpdir, file_name='obj2', contents=b'obj2-1', mtime=ORIG_MTIME - 1)
        self.CreateTempFile(tmpdir=tmpdir, file_name='obj3', contents=b'obj3-newer', mtime=ORIG_MTIME - 1)
        self.CreateTempFile(tmpdir=tmpdir, file_name='obj4', contents=b'obj4-2', mtime=ORIG_MTIME)
        self.CreateTempFile(tmpdir=tmpdir, file_name='obj5', contents=b'obj5-bigger', mtime=ORIG_MTIME)
        self.CreateTempFile(tmpdir=tmpdir, file_name='obj6', contents=b'obj6-1', mtime=ORIG_MTIME + 1)

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check():
            self.RunGsUtil(['rsync', '-u', tmpdir, suri(dst_bucket)])
            self.assertEqual('obj1-1', self.RunGsUtil(['cat', suri(dst_bucket, 'obj1')], return_stdout=True))
            self.assertEqual('obj2-1', self.RunGsUtil(['cat', suri(dst_bucket, 'obj2')], return_stdout=True))
            self.assertEqual('obj3-1', self.RunGsUtil(['cat', suri(dst_bucket, 'obj3')], return_stdout=True))
            self.assertEqual('obj4-1', self.RunGsUtil(['cat', suri(dst_bucket, 'obj4')], return_stdout=True))
            self.assertEqual('obj5-bigger', self.RunGsUtil(['cat', suri(dst_bucket, 'obj5')], return_stdout=True))
            self._VerifyObjectMtime(dst_bucket.bucket_name, 'obj6', str(ORIG_MTIME + 1))
        _Check()

    def test_dir_to_bucket_minus_i(self):
        """Tests that rsync -i works correctly."""
        tmpdir = self.CreateTempDir()
        dst_bucket = self.CreateBucket()
        ORIG_MTIME = 10
        self.CreateObject(bucket_uri=dst_bucket, object_name='obj1', contents=b'obj1-1')
        self.CreateObject(bucket_uri=dst_bucket, object_name='obj2', contents=b'obj2-1')
        self.CreateObject(bucket_uri=dst_bucket, object_name='obj3', contents=b'obj3-1', mtime=ORIG_MTIME)
        self.CreateTempFile(tmpdir=tmpdir, file_name='obj1', contents=b'obj1-2')
        self.CreateTempFile(tmpdir=tmpdir, file_name='obj2', contents=b'obj2-1', mtime=ORIG_MTIME)
        self.CreateTempFile(tmpdir=tmpdir, file_name='obj3', contents=b'obj3-newer', mtime=ORIG_MTIME - 1)

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check():
            self.RunGsUtil(['rsync', '-i', tmpdir, suri(dst_bucket)])
            self.assertEqual('obj1-1', self.RunGsUtil(['cat', suri(dst_bucket, 'obj1')], return_stdout=True))
            self.assertEqual('obj2-1', self.RunGsUtil(['cat', suri(dst_bucket, 'obj2')], return_stdout=True))
            self.assertEqual('obj3-1', self.RunGsUtil(['cat', suri(dst_bucket, 'obj3')], return_stdout=True))
        _Check()

    def test_rsync_files_with_whitespace(self):
        """Test to ensure filenames with whitespace can be rsynced"""
        filename = 'foo bar baz.txt'
        local_uris = []
        bucket_uri = self.CreateBucket()
        tmpdir = self.CreateTempDir()
        contents = 'File from rsync test: test_rsync_files_with_whitespace'
        local_file = self.CreateTempFile(tmpdir, contents, filename)
        expected_list_results = frozenset(['/foo bar baz.txt'])
        self.RunGsUtil(['rsync', '-r', tmpdir, suri(bucket_uri)])
        listing1 = TailSet(tmpdir, self.FlatListDir(tmpdir))
        listing2 = TailSet(suri(bucket_uri), self.FlatListBucket(bucket_uri))
        self.assertEqual(set(listing1), expected_list_results)
        self.assertEqual(set(listing2), expected_list_results)

    def test_rsync_files_with_special_characters(self):
        """Test to ensure filenames with special characters can be rsynced"""
        filename = 'Æ.txt'
        local_uris = []
        bucket_uri = self.CreateBucket()
        tmpdir = self.CreateTempDir()
        contents = 'File from rsync test: test_rsync_files_with_special_characters'
        local_file = self.CreateTempFile(tmpdir, contents, filename)
        expected_list_results = frozenset(['/Æ.txt'])
        self.RunGsUtil(['rsync', '-r', tmpdir, suri(bucket_uri)])
        listing1 = TailSet(tmpdir, self.FlatListDir(tmpdir))
        listing2 = TailSet(suri(bucket_uri), self.FlatListBucket(bucket_uri))
        self.assertEqual(set(listing1), expected_list_results)
        self.assertEqual(set(listing2), expected_list_results)

    @SkipForS3('No compressed transport encoding support for S3.')
    @SkipForXML('No compressed transport encoding support for the XML API.')
    @SequentialAndParallelTransfer
    def test_gzip_transport_encoded_all_upload(self):
        """Test gzip encoded files upload correctly."""
        file_names = ('test', 'test.txt', 'test.xml')
        local_uris = []
        bucket_uri = self.CreateBucket()
        tmpdir = self.CreateTempDir()
        contents = b'x' * 10000
        for file_name in file_names:
            local_uris.append(self.CreateTempFile(tmpdir, contents, file_name))
        stderr = self.RunGsUtil(['-D', 'rsync', '-J', '-r', tmpdir, suri(bucket_uri)], return_stderr=True)
        self.AssertNObjectsInBucket(bucket_uri, len(local_uris))
        for local_uri in local_uris:
            self.assertIn('Using compressed transport encoding for file://%s.' % local_uri, stderr)
        if not self._use_gcloud_storage:
            self.assertIn('send: Using gzip transport encoding for the request.', stderr)

    @SkipForS3('No compressed transport encoding support for S3.')
    @SkipForXML('No compressed transport encoding support for the XML API.')
    @SequentialAndParallelTransfer
    def test_gzip_transport_encoded_filtered_upload(self):
        """Test gzip encoded files upload correctly."""
        file_names_valid = ('test.txt', 'photo.txt')
        file_names_invalid = ('file', 'test.png', 'test.xml')
        local_uris_valid = []
        local_uris_invalid = []
        bucket_uri = self.CreateBucket()
        tmpdir = self.CreateTempDir()
        contents = b'x' * 10000
        for file_name in file_names_valid:
            local_uris_valid.append(self.CreateTempFile(tmpdir, contents, file_name))
        for file_name in file_names_invalid:
            local_uris_invalid.append(self.CreateTempFile(tmpdir, contents, file_name))
        stderr = self.RunGsUtil(['-D', 'rsync', '-j', 'txt', '-r', tmpdir, suri(bucket_uri)], return_stderr=True)
        self.AssertNObjectsInBucket(bucket_uri, len(file_names_valid) + len(file_names_invalid))
        for local_uri in local_uris_valid:
            self.assertIn('Using compressed transport encoding for file://%s.' % local_uri, stderr)
        for local_uri in local_uris_invalid:
            self.assertNotIn('Using compressed transport encoding for file://%s.' % local_uri, stderr)
        if not self._use_gcloud_storage:
            self.assertIn('send: Using gzip transport encoding for the request.', stderr)

    @SkipForS3('No compressed transport encoding support for S3.')
    @SkipForXML('No compressed transport encoding support for the XML API.')
    @SequentialAndParallelTransfer
    def test_gzip_transport_encoded_all_upload_parallel(self):
        """Test gzip encoded files upload correctly."""
        file_names = ('test', 'test.txt', 'test.xml')
        local_uris = []
        bucket_uri = self.CreateBucket()
        tmpdir = self.CreateTempDir()
        contents = b'x' * 10000
        for file_name in file_names:
            local_uris.append(self.CreateTempFile(tmpdir, contents, file_name))
        stderr = self.RunGsUtil(['-D', '-m', 'rsync', '-J', '-r', tmpdir, suri(bucket_uri)], return_stderr=True)
        self.AssertNObjectsInBucket(bucket_uri, len(local_uris))
        for local_uri in local_uris:
            self.assertIn('Using compressed transport encoding for file://%s.' % local_uri, stderr)
        if not self._use_gcloud_storage:
            self.assertIn('send: Using gzip transport encoding for the request.', stderr)

    @SkipForS3('Test uses gs-specific KMS encryption')
    def test_kms_key_applied_to_dest_objects(self):
        bucket_uri = self.CreateBucket()
        cloud_container_suri = suri(bucket_uri) + '/foo'
        obj_name = 'bar'
        obj_contents = b'bar'
        tmp_dir = self.CreateTempDir()
        self.CreateTempFile(tmpdir=tmp_dir, file_name=obj_name, contents=obj_contents)
        key_fqn = AuthorizeProjectToUseTestingKmsKey()
        with SetBotoConfigForTest([('GSUtil', 'encryption_key', key_fqn)]):
            self.RunGsUtil(['rsync', tmp_dir, cloud_container_suri])
        with SetBotoConfigForTest([('GSUtil', 'prefer_api', 'json')]):
            stdout = self.RunGsUtil(['ls', '-L', '%s/%s' % (cloud_container_suri, obj_name)], return_stdout=True)
        self.assertRegex(stdout, 'KMS key:\\s+%s' % key_fqn)

    @SkipForGS('Tests that gs-specific encryption settings are skipped for s3.')
    def test_kms_key_specified_will_not_prevent_non_kms_copy_to_s3(self):
        tmp_dir = self.CreateTempDir()
        self.CreateTempFile(tmpdir=tmp_dir, contents=b'foo')
        bucket_uri = self.CreateBucket()
        dummy_key = 'projects/myproject/locations/global/keyRings/mykeyring/cryptoKeys/mykey'
        with SetBotoConfigForTest([('GSUtil', 'prefer_api', 'json')]):
            self.RunGsUtil(['rsync', tmp_dir, suri(bucket_uri)])

    def test_bucket_to_bucket_includes_arbitrary_headers(self):
        bucket1_uri = self.CreateBucket()
        bucket2_uri = self.CreateBucket()
        self.CreateObject(bucket_uri=bucket1_uri, object_name='obj1', contents=b'obj1')
        stderr = self.RunGsUtil(['-DD', '-h', 'arbitrary:header', 'rsync', suri(bucket1_uri), suri(bucket2_uri)], return_stderr=True)
        if self._use_gcloud_storage:
            request_count = len(re.findall('= request start =', stderr))
            target_header_count = len(re.findall("b'arbitrary': b'header'", stderr))
            self.assertEqual(request_count, target_header_count)
        else:
            headers_for_all_requests = re.findall('Headers: \\{([\\s\\S]*?)\\}', stderr)
            self.assertTrue(headers_for_all_requests)
            for headers in headers_for_all_requests:
                self.assertIn("'arbitrary': 'header'", headers)

    @SkipForGS('Tests that S3 SSE-C is handled.')
    def test_s3_sse_is_handled_with_arbitrary_headers(self):
        tmp_dir = self.CreateTempDir()
        tmp_file = self.CreateTempFile(tmpdir=tmp_dir, contents=b'foo')
        bucket_uri1 = self.CreateBucket()
        bucket_uri2 = self.CreateBucket()
        header_flags = ['-h', '"x-amz-server-side-encryption-customer-algorithm:AES256"', '-h', '"x-amz-server-side-encryption-customer-key:{}"'.format(TEST_ENCRYPTION_KEY_S3), '-h', '"x-amz-server-side-encryption-customer-key-md5:{}"'.format(TEST_ENCRYPTION_KEY_S3_MD5)]
        with SetBotoConfigForTest([('GSUtil', 'check_hashes', 'never')]):
            self.RunGsUtil(header_flags + ['cp', tmp_file, suri(bucket_uri1, 'test')])
            self.RunGsUtil(header_flags + ['rsync', suri(bucket_uri1), suri(bucket_uri2)])
            contents = self.RunGsUtil(header_flags + ['cat', suri(bucket_uri2, 'test')], return_stdout=True)
        self.assertEqual(contents, 'foo')