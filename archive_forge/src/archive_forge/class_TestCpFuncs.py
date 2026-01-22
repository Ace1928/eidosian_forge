from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import collections
import datetime
import logging
import os
import pyu2f
from apitools.base.py import exceptions as apitools_exceptions
from gslib.bucket_listing_ref import BucketListingObject
from gslib.bucket_listing_ref import BucketListingPrefix
from gslib.cloud_api import CloudApi
from gslib.cloud_api import ResumableUploadAbortException
from gslib.cloud_api import ResumableUploadException
from gslib.cloud_api import ResumableUploadStartOverException
from gslib.cloud_api import ServiceException
from gslib.command import CreateOrGetGsutilLogger
from gslib.discard_messages_queue import DiscardMessagesQueue
from gslib.exception import CommandException
from gslib.gcs_json_api import GcsJsonApi
from gslib.parallel_tracker_file import ObjectFromTracker
from gslib.storage_url import StorageUrlFromString
from gslib.tests.mock_cloud_api import MockCloudApi
from gslib.tests.testcase.unit_testcase import GsUtilUnitTestCase
from gslib.tests.util import GSMockBucketStorageUri
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import unittest
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
from gslib.utils import copy_helper
from gslib.utils import parallelism_framework_util
from gslib.utils import posix_util
from gslib.utils import system_util
from gslib.utils import hashing_helper
from gslib.utils.copy_helper import _CheckCloudHashes
from gslib.utils.copy_helper import _DelegateUploadFileToObject
from gslib.utils.copy_helper import _GetPartitionInfo
from gslib.utils.copy_helper import _SelectUploadCompressionStrategy
from gslib.utils.copy_helper import _SetContentTypeFromFile
from gslib.utils.copy_helper import ExpandUrlToSingleBlr
from gslib.utils.copy_helper import FilterExistingComponents
from gslib.utils.copy_helper import GZIP_ALL_FILES
from gslib.utils.copy_helper import PerformParallelUploadFileToObjectArgs
from gslib.utils.copy_helper import WarnIfMvEarlyDeletionChargeApplies
from six import add_move, MovedModule
from six.moves import mock
class TestCpFuncs(GsUtilUnitTestCase):
    """Unit tests for parallel upload functions in cp command."""

    def testGetPartitionInfo(self):
        """Tests the _GetPartitionInfo function."""
        num_components, component_size = _GetPartitionInfo(300, 200, 10)
        self.assertEqual(30, num_components)
        self.assertEqual(10, component_size)
        num_components, component_size = _GetPartitionInfo(301, 200, 10)
        self.assertEqual(31, num_components)
        self.assertEqual(10, component_size)
        num_components, component_size = _GetPartitionInfo(299, 200, 10)
        self.assertEqual(30, num_components)
        self.assertEqual(10, component_size)
        num_components, component_size = _GetPartitionInfo(301, 2, 10)
        self.assertEqual(2, num_components)
        self.assertEqual(151, component_size)
        num_components, component_size = _GetPartitionInfo(10 ** 150 + 1, 10 ** 200, 10)
        self.assertEqual(10 ** 149 + 1, num_components)
        self.assertEqual(10, component_size)
        num_components, component_size = _GetPartitionInfo(10 ** 150 + 1, 10, 10)
        self.assertEqual(10, num_components)
        self.assertEqual(10 ** 149 + 1, component_size)
        num_components, component_size = _GetPartitionInfo(100, 500, 51)
        self.assertEqual(2, num_components)
        self.assertEqual(50, component_size)

    def testFilterExistingComponentsNonVersioned(self):
        """Tests upload with a variety of component states."""
        mock_api = MockCloudApi()
        bucket_name = self.MakeTempName('bucket')
        tracker_file = self.CreateTempFile(file_name='foo', contents=b'asdf')
        tracker_file_lock = parallelism_framework_util.CreateLock()
        content_type = 'ContentType'
        storage_class = 'StorageClass'
        fpath_uploaded_correctly = self.CreateTempFile(file_name='foo1', contents=b'1')
        fpath_uploaded_correctly_url = StorageUrlFromString(str(fpath_uploaded_correctly))
        object_uploaded_correctly_url = StorageUrlFromString('%s://%s/%s' % (self.default_provider, bucket_name, fpath_uploaded_correctly))
        with open(fpath_uploaded_correctly, 'rb') as f_in:
            fpath_uploaded_correctly_md5 = _CalculateB64EncodedMd5FromContents(f_in)
        mock_api.MockCreateObjectWithMetadata(apitools_messages.Object(bucket=bucket_name, name=fpath_uploaded_correctly, md5Hash=fpath_uploaded_correctly_md5), contents=b'1')
        args_uploaded_correctly = PerformParallelUploadFileToObjectArgs(fpath_uploaded_correctly, 0, 1, fpath_uploaded_correctly_url, object_uploaded_correctly_url, '', content_type, storage_class, tracker_file, tracker_file_lock, None, False)
        fpath_not_uploaded = self.CreateTempFile(file_name='foo2', contents=b'2')
        fpath_not_uploaded_url = StorageUrlFromString(str(fpath_not_uploaded))
        object_not_uploaded_url = StorageUrlFromString('%s://%s/%s' % (self.default_provider, bucket_name, fpath_not_uploaded))
        args_not_uploaded = PerformParallelUploadFileToObjectArgs(fpath_not_uploaded, 0, 1, fpath_not_uploaded_url, object_not_uploaded_url, '', content_type, storage_class, tracker_file, tracker_file_lock, None, False)
        fpath_wrong_contents = self.CreateTempFile(file_name='foo4', contents=b'4')
        fpath_wrong_contents_url = StorageUrlFromString(str(fpath_wrong_contents))
        object_wrong_contents_url = StorageUrlFromString('%s://%s/%s' % (self.default_provider, bucket_name, fpath_wrong_contents))
        with open(self.CreateTempFile(contents=b'_'), 'rb') as f_in:
            fpath_wrong_contents_md5 = _CalculateB64EncodedMd5FromContents(f_in)
        mock_api.MockCreateObjectWithMetadata(apitools_messages.Object(bucket=bucket_name, name=fpath_wrong_contents, md5Hash=fpath_wrong_contents_md5), contents=b'1')
        args_wrong_contents = PerformParallelUploadFileToObjectArgs(fpath_wrong_contents, 0, 1, fpath_wrong_contents_url, object_wrong_contents_url, '', content_type, storage_class, tracker_file, tracker_file_lock, None, False)
        fpath_remote_deleted = self.CreateTempFile(file_name='foo5', contents=b'5')
        fpath_remote_deleted_url = StorageUrlFromString(str(fpath_remote_deleted))
        args_remote_deleted = PerformParallelUploadFileToObjectArgs(fpath_remote_deleted, 0, 1, fpath_remote_deleted_url, '', '', content_type, storage_class, tracker_file, tracker_file_lock, None, False)
        fpath_no_longer_used = self.CreateTempFile(file_name='foo6', contents=b'6')
        with open(fpath_no_longer_used, 'rb') as f_in:
            file_md5 = _CalculateB64EncodedMd5FromContents(f_in)
        mock_api.MockCreateObjectWithMetadata(apitools_messages.Object(bucket=bucket_name, name='foo6', md5Hash=file_md5), contents=b'6')
        dst_args = {fpath_uploaded_correctly: args_uploaded_correctly, fpath_not_uploaded: args_not_uploaded, fpath_wrong_contents: args_wrong_contents, fpath_remote_deleted: args_remote_deleted}
        existing_components = [ObjectFromTracker(fpath_uploaded_correctly, ''), ObjectFromTracker(fpath_wrong_contents, ''), ObjectFromTracker(fpath_remote_deleted, ''), ObjectFromTracker(fpath_no_longer_used, '')]
        bucket_url = StorageUrlFromString('%s://%s' % (self.default_provider, bucket_name))
        components_to_upload, uploaded_components, existing_objects_to_delete = FilterExistingComponents(dst_args, existing_components, bucket_url, mock_api)
        uploaded_components = [i[0] for i in uploaded_components]
        for arg in [args_not_uploaded, args_wrong_contents, args_remote_deleted]:
            self.assertTrue(arg in components_to_upload)
        self.assertEqual(1, len(uploaded_components))
        self.assertEqual(args_uploaded_correctly.dst_url.url_string, uploaded_components[0].url_string)
        self.assertEqual(1, len(existing_objects_to_delete))
        no_longer_used_url = StorageUrlFromString('%s://%s/%s' % (self.default_provider, bucket_name, fpath_no_longer_used))
        self.assertEqual(no_longer_used_url.url_string, existing_objects_to_delete[0].url_string)

    def testFilterExistingComponentsVersioned(self):
        """Tests upload with versionined parallel components."""
        mock_api = MockCloudApi()
        bucket_name = self.MakeTempName('bucket')
        mock_api.MockCreateVersionedBucket(bucket_name)
        content_type = 'ContentType'
        storage_class = 'StorageClass'
        tracker_file = self.CreateTempFile(file_name='foo', contents=b'asdf')
        tracker_file_lock = parallelism_framework_util.CreateLock()
        fpath_uploaded_correctly = self.CreateTempFile(file_name='foo1', contents=b'1')
        fpath_uploaded_correctly_url = StorageUrlFromString(str(fpath_uploaded_correctly))
        with open(fpath_uploaded_correctly, 'rb') as f_in:
            fpath_uploaded_correctly_md5 = _CalculateB64EncodedMd5FromContents(f_in)
        object_uploaded_correctly = mock_api.MockCreateObjectWithMetadata(apitools_messages.Object(bucket=bucket_name, name=fpath_uploaded_correctly, md5Hash=fpath_uploaded_correctly_md5), contents=b'1')
        object_uploaded_correctly_url = StorageUrlFromString('%s://%s/%s#%s' % (self.default_provider, bucket_name, fpath_uploaded_correctly, object_uploaded_correctly.generation))
        args_uploaded_correctly = PerformParallelUploadFileToObjectArgs(fpath_uploaded_correctly, 0, 1, fpath_uploaded_correctly_url, object_uploaded_correctly_url, object_uploaded_correctly.generation, content_type, storage_class, tracker_file, tracker_file_lock, None, False)
        fpath_duplicate = fpath_uploaded_correctly
        fpath_duplicate_url = StorageUrlFromString(str(fpath_duplicate))
        duplicate_uploaded_correctly = mock_api.MockCreateObjectWithMetadata(apitools_messages.Object(bucket=bucket_name, name=fpath_duplicate, md5Hash=fpath_uploaded_correctly_md5), contents=b'1')
        duplicate_uploaded_correctly_url = StorageUrlFromString('%s://%s/%s#%s' % (self.default_provider, bucket_name, fpath_uploaded_correctly, duplicate_uploaded_correctly.generation))
        args_duplicate = PerformParallelUploadFileToObjectArgs(fpath_duplicate, 0, 1, fpath_duplicate_url, duplicate_uploaded_correctly_url, duplicate_uploaded_correctly.generation, content_type, storage_class, tracker_file, tracker_file_lock, None, False)
        fpath_wrong_contents = self.CreateTempFile(file_name='foo4', contents=b'4')
        fpath_wrong_contents_url = StorageUrlFromString(str(fpath_wrong_contents))
        with open(self.CreateTempFile(contents=b'_'), 'rb') as f_in:
            fpath_wrong_contents_md5 = _CalculateB64EncodedMd5FromContents(f_in)
        object_wrong_contents = mock_api.MockCreateObjectWithMetadata(apitools_messages.Object(bucket=bucket_name, name=fpath_wrong_contents, md5Hash=fpath_wrong_contents_md5), contents=b'_')
        wrong_contents_url = StorageUrlFromString('%s://%s/%s#%s' % (self.default_provider, bucket_name, fpath_wrong_contents, object_wrong_contents.generation))
        args_wrong_contents = PerformParallelUploadFileToObjectArgs(fpath_wrong_contents, 0, 1, fpath_wrong_contents_url, wrong_contents_url, '', content_type, storage_class, tracker_file, tracker_file_lock, None, False)
        dst_args = {fpath_uploaded_correctly: args_uploaded_correctly, fpath_wrong_contents: args_wrong_contents}
        existing_components = [ObjectFromTracker(fpath_uploaded_correctly, object_uploaded_correctly_url.generation), ObjectFromTracker(fpath_duplicate, duplicate_uploaded_correctly_url.generation), ObjectFromTracker(fpath_wrong_contents, wrong_contents_url.generation)]
        bucket_url = StorageUrlFromString('%s://%s' % (self.default_provider, bucket_name))
        components_to_upload, uploaded_components, existing_objects_to_delete = FilterExistingComponents(dst_args, existing_components, bucket_url, mock_api)
        uploaded_components = [i[0] for i in uploaded_components]
        self.assertEqual([args_wrong_contents], components_to_upload)
        self.assertEqual(args_uploaded_correctly.dst_url.url_string, uploaded_components[0].url_string)
        expected_to_delete = [(args_wrong_contents.dst_url.object_name, args_wrong_contents.dst_url.generation), (args_duplicate.dst_url.object_name, args_duplicate.dst_url.generation)]
        for uri in existing_objects_to_delete:
            self.assertTrue((uri.object_name, uri.generation) in expected_to_delete)
        self.assertEqual(len(expected_to_delete), len(existing_objects_to_delete))

    def testReauthChallengeIsPerformed(self):
        mock_api = mock.Mock(spec=CloudApi)
        destination_url = StorageUrlFromString('gs://bucket')
        with SetBotoConfigForTest([('GSUtil', 'trigger_reauth_challenge_for_parallel_operations', 'True')]):
            copy_helper.TriggerReauthForDestinationProviderIfNecessary(destination_url, mock_api, worker_count=2)
        mock_api.GetBucket.assert_called_once_with('bucket', fields=['location'], provider='gs')

    def testReauthChallengeIsNotPerformedByDefault(self):
        mock_api = mock.Mock(spec=CloudApi)
        destination_url = StorageUrlFromString('gs://bucket')
        copy_helper.TriggerReauthForDestinationProviderIfNecessary(destination_url, mock_api, worker_count=2)
        mock_api.GetBucket.assert_not_called()

    def testReauthChallengeNotPerformedWithFileDestination(self):
        mock_api = mock.Mock(spec=CloudApi)
        destination_url = StorageUrlFromString('dir/file')
        with SetBotoConfigForTest([('GSUtil', 'trigger_reauth_challenge_for_parallel_operations', 'True')]):
            copy_helper.TriggerReauthForDestinationProviderIfNecessary(destination_url, mock_api, worker_count=2)
        mock_api.GetBucket.assert_not_called()

    def testReauthChallengeNotPerformedWhenDestinationContainsWildcard(self):
        mock_api = mock.Mock(spec=CloudApi)
        destination_url = StorageUrlFromString('gs://bucket*')
        with SetBotoConfigForTest([('GSUtil', 'trigger_reauth_challenge_for_parallel_operations', 'True')]):
            copy_helper.TriggerReauthForDestinationProviderIfNecessary(destination_url, mock_api, worker_count=2)
        mock_api.GetBucket.assert_not_called()

    def testReauthChallengeNotPerformedWithSequentialExecution(self):
        mock_api = mock.Mock(spec=CloudApi)
        destination_url = StorageUrlFromString('gs://bucket')
        with SetBotoConfigForTest([('GSUtil', 'trigger_reauth_challenge_for_parallel_operations', 'True')]):
            copy_helper.TriggerReauthForDestinationProviderIfNecessary(destination_url, mock_api, worker_count=1)
        mock_api.GetBucket.assert_not_called()

    def testReauthChallengeRaisesReauthError(self):
        mock_api = mock.Mock(spec=CloudApi)
        mock_api.GetBucket.side_effect = pyu2f.errors.PluginError('Reauth error')
        destination_url = StorageUrlFromString('gs://bucket')
        with SetBotoConfigForTest([('GSUtil', 'trigger_reauth_challenge_for_parallel_operations', 'True')]):
            with self.assertRaisesRegex(pyu2f.errors.PluginError, 'Reauth error'):
                copy_helper.TriggerReauthForDestinationProviderIfNecessary(destination_url, mock_api, worker_count=2)

    def testReauthChallengeSilencesOtherErrors(self):
        mock_api = mock.Mock(spec=CloudApi)
        mock_api.GetBucket.side_effect = Exception
        destination_url = StorageUrlFromString('gs://bucket')
        with SetBotoConfigForTest([('GSUtil', 'trigger_reauth_challenge_for_parallel_operations', 'True')]):
            copy_helper.TriggerReauthForDestinationProviderIfNecessary(destination_url, mock_api, worker_count=2)
        mock_api.GetBucket.assert_called_once_with('bucket', fields=['location'], provider='gs')

    def testTranslateApitoolsResumableUploadException(self):
        """Tests that _TranslateApitoolsResumableUploadException works correctly."""
        gsutil_api = GcsJsonApi(GSMockBucketStorageUri, CreateOrGetGsutilLogger('copy_test'), DiscardMessagesQueue())
        gsutil_api.http.disable_ssl_certificate_validation = True
        exc = apitools_exceptions.HttpError({'status': 503}, None, None)
        translated_exc = gsutil_api._TranslateApitoolsResumableUploadException(exc)
        self.assertTrue(isinstance(translated_exc, ServiceException))
        gsutil_api.http.disable_ssl_certificate_validation = False
        exc = apitools_exceptions.HttpError({'status': 503}, None, None)
        translated_exc = gsutil_api._TranslateApitoolsResumableUploadException(exc)
        self.assertTrue(isinstance(translated_exc, ResumableUploadException))
        gsutil_api.http.disable_ssl_certificate_validation = False
        exc = apitools_exceptions.HttpError({'status': 429}, None, None)
        translated_exc = gsutil_api._TranslateApitoolsResumableUploadException(exc)
        self.assertTrue(isinstance(translated_exc, ResumableUploadException))
        exc = apitools_exceptions.HttpError({'status': 410}, None, None)
        translated_exc = gsutil_api._TranslateApitoolsResumableUploadException(exc)
        self.assertTrue(isinstance(translated_exc, ResumableUploadStartOverException))
        exc = apitools_exceptions.HttpError({'status': 404}, None, None)
        translated_exc = gsutil_api._TranslateApitoolsResumableUploadException(exc)
        self.assertTrue(isinstance(translated_exc, ResumableUploadStartOverException))
        exc = apitools_exceptions.HttpError({'status': 401}, None, None)
        translated_exc = gsutil_api._TranslateApitoolsResumableUploadException(exc)
        self.assertTrue(isinstance(translated_exc, ResumableUploadAbortException))
        exc = apitools_exceptions.TransferError('Aborting transfer')
        translated_exc = gsutil_api._TranslateApitoolsResumableUploadException(exc)
        self.assertTrue(isinstance(translated_exc, ResumableUploadAbortException))
        exc = apitools_exceptions.TransferError('additional bytes left in stream')
        translated_exc = gsutil_api._TranslateApitoolsResumableUploadException(exc)
        self.assertTrue(isinstance(translated_exc, ResumableUploadAbortException))
        self.assertIn('this can happen if a file changes size', translated_exc.reason)

    def testTranslateApitoolsResumableUploadExceptionStreamExhausted(self):
        """Test that StreamExhausted error gets handled."""
        gsutil_api = GcsJsonApi(GSMockBucketStorageUri, CreateOrGetGsutilLogger('copy_test'), DiscardMessagesQueue())
        exc = apitools_exceptions.StreamExhausted('Not enough bytes')
        translated_exc = gsutil_api._TranslateApitoolsResumableUploadException(exc)
        self.assertTrue(isinstance(translated_exc, ResumableUploadAbortException))
        self.assertIn('if this issue persists, try deleting the tracker files present under ~/.gsutil/tracker-files/', translated_exc.reason)

    def testSetContentTypeFromFile(self):
        """Tests that content type is correctly determined for symlinks."""
        if system_util.IS_WINDOWS:
            return unittest.skip('use_magicfile features not available on Windows')
        surprise_html = b'<html><body>And you thought I was just text!</body></html>'
        temp_dir_path = self.CreateTempDir()
        txt_file_path = self.CreateTempFile(tmpdir=temp_dir_path, contents=surprise_html, file_name='html_in_disguise.txt')
        link_name = 'link_to_realfile'
        os.symlink(txt_file_path, temp_dir_path + os.path.sep + link_name)
        dst_obj_metadata_mock = mock.MagicMock(contentType=None)
        src_url_stub = mock.MagicMock(object_name=temp_dir_path + os.path.sep + link_name)
        src_url_stub.IsFileUrl.return_value = True
        src_url_stub.IsStream.return_value = False
        src_url_stub.IsFifo.return_value = False
        with SetBotoConfigForTest([('GSUtil', 'use_magicfile', 'True')]):
            _SetContentTypeFromFile(src_url_stub, dst_obj_metadata_mock)
        self.assertEqual('text/html; charset=us-ascii', dst_obj_metadata_mock.contentType)
        dst_obj_metadata_mock = mock.MagicMock(contentType=None)
        with SetBotoConfigForTest([('GSUtil', 'use_magicfile', 'False')]):
            _SetContentTypeFromFile(src_url_stub, dst_obj_metadata_mock)
        self.assertEqual('text/plain', dst_obj_metadata_mock.contentType)

    def testSetsContentTypesForCommonFileExtensionsCorrectly(self):
        extension_rules = copy_helper.COMMON_EXTENSION_RULES.items()
        for extension, expected_content_type in extension_rules:
            dst_obj_metadata_mock = mock.MagicMock(contentType=None)
            src_url_stub = mock.MagicMock(object_name='file.' + extension)
            src_url_stub.IsFileUrl.return_value = True
            src_url_stub.IsStream.return_value = False
            src_url_stub.IsFifo.return_value = False
            _SetContentTypeFromFile(src_url_stub, dst_obj_metadata_mock)
            self.assertEqual(expected_content_type, dst_obj_metadata_mock.contentType)
    _PI_DAY = datetime.datetime(2016, 3, 14, 15, 9, 26)

    @mock.patch('time.time', new=mock.MagicMock(return_value=posix_util.ConvertDatetimeToPOSIX(_PI_DAY)))
    def testWarnIfMvEarlyDeletionChargeApplies(self):
        """Tests that WarnIfEarlyDeletionChargeApplies warns when appropriate."""
        test_logger = logging.Logger('test')
        src_url = StorageUrlFromString('gs://bucket/object')
        for object_time_created in (self._PI_DAY, self._PI_DAY - datetime.timedelta(days=29, hours=23)):
            recent_nearline_obj = apitools_messages.Object(storageClass='NEARLINE', timeCreated=object_time_created)
            with mock.patch.object(test_logger, 'warn') as mocked_warn:
                WarnIfMvEarlyDeletionChargeApplies(src_url, recent_nearline_obj, test_logger)
                mocked_warn.assert_called_with('Warning: moving %s object %s may incur an early deletion charge, because the original object is less than %s days old according to the local system time.', 'nearline', src_url.url_string, 30)
        for object_time_created in (self._PI_DAY, self._PI_DAY - datetime.timedelta(days=89, hours=23)):
            recent_nearline_obj = apitools_messages.Object(storageClass='COLDLINE', timeCreated=object_time_created)
            with mock.patch.object(test_logger, 'warn') as mocked_warn:
                WarnIfMvEarlyDeletionChargeApplies(src_url, recent_nearline_obj, test_logger)
                mocked_warn.assert_called_with('Warning: moving %s object %s may incur an early deletion charge, because the original object is less than %s days old according to the local system time.', 'coldline', src_url.url_string, 90)
        for object_time_created in (self._PI_DAY, self._PI_DAY - datetime.timedelta(days=364, hours=23)):
            recent_archive_obj = apitools_messages.Object(storageClass='ARCHIVE', timeCreated=object_time_created)
            with mock.patch.object(test_logger, 'warn') as mocked_warn:
                WarnIfMvEarlyDeletionChargeApplies(src_url, recent_archive_obj, test_logger)
                mocked_warn.assert_called_with('Warning: moving %s object %s may incur an early deletion charge, because the original object is less than %s days old according to the local system time.', 'archive', src_url.url_string, 365)
        with mock.patch.object(test_logger, 'warn') as mocked_warn:
            old_nearline_obj = apitools_messages.Object(storageClass='NEARLINE', timeCreated=self._PI_DAY - datetime.timedelta(days=30, seconds=1))
            WarnIfMvEarlyDeletionChargeApplies(src_url, old_nearline_obj, test_logger)
            mocked_warn.assert_not_called()
        with mock.patch.object(test_logger, 'warn') as mocked_warn:
            old_coldline_obj = apitools_messages.Object(storageClass='COLDLINE', timeCreated=self._PI_DAY - datetime.timedelta(days=90, seconds=1))
            WarnIfMvEarlyDeletionChargeApplies(src_url, old_coldline_obj, test_logger)
            mocked_warn.assert_not_called()
        with mock.patch.object(test_logger, 'warn') as mocked_warn:
            old_archive_obj = apitools_messages.Object(storageClass='ARCHIVE', timeCreated=self._PI_DAY - datetime.timedelta(days=365, seconds=1))
            WarnIfMvEarlyDeletionChargeApplies(src_url, old_archive_obj, test_logger)
            mocked_warn.assert_not_called()
        with mock.patch.object(test_logger, 'warn') as mocked_warn:
            not_old_enough_nearline_obj = apitools_messages.Object(storageClass='STANDARD', timeCreated=self._PI_DAY)
            WarnIfMvEarlyDeletionChargeApplies(src_url, not_old_enough_nearline_obj, test_logger)
            mocked_warn.assert_not_called()

    def testSelectUploadCompressionStrategyAll(self):
        paths = ('file://test', 'test.xml', 'test.py')
        exts = GZIP_ALL_FILES
        for path in paths:
            zipped, gzip_encoded = _SelectUploadCompressionStrategy(path, False, exts, False)
            self.assertTrue(zipped)
            self.assertFalse(gzip_encoded)
            zipped, gzip_encoded = _SelectUploadCompressionStrategy(path, False, exts, True)
            self.assertFalse(zipped)
            self.assertTrue(gzip_encoded)

    def testSelectUploadCompressionStrategyFilter(self):
        zipped, gzip_encoded = _SelectUploadCompressionStrategy('test.xml', False, ['xml'], False)
        self.assertTrue(zipped)
        self.assertFalse(gzip_encoded)
        zipped, gzip_encoded = _SelectUploadCompressionStrategy('test.xml', False, ['yaml'], False)
        self.assertFalse(zipped)
        self.assertFalse(gzip_encoded)

    def testSelectUploadCompressionStrategyComponent(self):
        zipped, gzip_encoded = _SelectUploadCompressionStrategy('test.xml', True, ['not_matching'], True)
        self.assertFalse(zipped)
        self.assertTrue(gzip_encoded)

    def testDelegateUploadFileToObjectNormal(self):
        mock_stream = mock.Mock()
        mock_stream.close = mock.Mock()

        def DelegateUpload():
            return ('a', 'b')
        elapsed_time, uploaded_object = _DelegateUploadFileToObject(DelegateUpload, 'url', mock_stream, False, False, False, None)
        self.assertEqual(elapsed_time, 'a')
        self.assertEqual(uploaded_object, 'b')
        self.assertTrue(mock_stream.close.called)

    @mock.patch('os.unlink')
    def testDelegateUploadFileToObjectZipped(self, mock_unlink):
        mock_stream = mock.Mock()
        mock_stream.close = mock.Mock()
        mock_upload_url = mock.Mock()
        mock_upload_url.object_name = 'Sample'

        def DelegateUpload():
            return ('a', 'b')
        elapsed_time, uploaded_object = _DelegateUploadFileToObject(DelegateUpload, mock_upload_url, mock_stream, True, False, False, None)
        self.assertEqual(elapsed_time, 'a')
        self.assertEqual(uploaded_object, 'b')
        self.assertTrue(mock_unlink.called)
        self.assertTrue(mock_stream.close.called)

    @mock.patch('gslib.command.concurrent_compressed_upload_lock')
    def testDelegateUploadFileToObjectGzipEncoded(self, mock_lock):
        mock_stream = mock.Mock()
        mock_stream.close = mock.Mock()

        def DelegateUpload():
            self.assertTrue(mock_lock.__enter__.called)
            return ('a', 'b')
        elapsed_time, uploaded_object = _DelegateUploadFileToObject(DelegateUpload, 'url', mock_stream, False, True, False, None)
        self.assertEqual(elapsed_time, 'a')
        self.assertEqual(uploaded_object, 'b')
        self.assertTrue(mock_stream.close.called)
        self.assertTrue(mock_lock.__exit__.called)

    @mock.patch('gslib.command.concurrent_compressed_upload_lock')
    def testDelegateUploadFileToObjectGzipEncodedComposite(self, mock_lock):
        mock_stream = mock.Mock()
        mock_stream.close = mock.Mock()

        def DelegateUpload():
            self.assertFalse(mock_lock.__enter__.called)
            return ('a', 'b')
        elapsed_time, uploaded_object = _DelegateUploadFileToObject(DelegateUpload, 'url', mock_stream, False, True, True, None)
        self.assertEqual(elapsed_time, 'a')
        self.assertEqual(uploaded_object, 'b')
        self.assertTrue(mock_stream.close.called)
        self.assertFalse(mock_lock.__exit__.called)

    def testDoesNotGetSizeSourceFieldIfFileSizeWillChange(self):
        fields = copy_helper.GetSourceFieldsNeededForCopy(True, True, False, file_size_will_change=True)
        self.assertNotIn('size', fields)