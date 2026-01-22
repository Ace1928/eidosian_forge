from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import ast
import base64
import binascii
import datetime
import gzip
import logging
import os
import pickle
import pkgutil
import random
import re
import stat
import string
import sys
import threading
from unittest import mock
from apitools.base.py import exceptions as apitools_exceptions
import boto
from boto import storage_uri
from boto.exception import ResumableTransferDisposition
from boto.exception import StorageResponseError
from boto.storage_uri import BucketStorageUri
from gslib import command
from gslib import exception
from gslib import name_expansion
from gslib.cloud_api import ResumableUploadStartOverException
from gslib.commands.config import DEFAULT_SLICED_OBJECT_DOWNLOAD_THRESHOLD
from gslib.commands.cp import ShimTranslatePredefinedAclSubOptForCopy
from gslib.cs_api_map import ApiSelector
from gslib.daisy_chain_wrapper import _DEFAULT_DOWNLOAD_CHUNK_SIZE
from gslib.discard_messages_queue import DiscardMessagesQueue
from gslib.exception import InvalidUrlError
from gslib.gcs_json_api import GcsJsonApi
from gslib.parallel_tracker_file import ObjectFromTracker
from gslib.parallel_tracker_file import WriteParallelUploadTrackerFile
from gslib.project_id import PopulateProjectId
from gslib.storage_url import StorageUrlFromString
from gslib.tests.rewrite_helper import EnsureRewriteResumeCallbackHandler
from gslib.tests.rewrite_helper import HaltingRewriteCallbackHandler
from gslib.tests.rewrite_helper import RewriteHaltException
import gslib.tests.testcase as testcase
from gslib.tests.testcase.base import NotParallelizable
from gslib.tests.testcase.integration_testcase import SkipForGS
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.testcase.integration_testcase import SkipForXML
from gslib.tests.testcase.integration_testcase import SkipForJSON
from gslib.tests.util import AuthorizeProjectToUseTestingKmsKey
from gslib.tests.util import BuildErrorRegex
from gslib.tests.util import GenerationFromURI as urigen
from gslib.tests.util import HaltingCopyCallbackHandler
from gslib.tests.util import HaltOneComponentCopyCallbackHandler
from gslib.tests.util import HAS_GS_PORT
from gslib.tests.util import HAS_S3_CREDS
from gslib.tests.util import KmsTestingResources
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
from gslib.tests.util import TEST_ENCRYPTION_KEY1
from gslib.tests.util import TEST_ENCRYPTION_KEY1_SHA256_B64
from gslib.tests.util import TEST_ENCRYPTION_KEY2
from gslib.tests.util import TEST_ENCRYPTION_KEY3
from gslib.tests.util import unittest
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
from gslib.tracker_file import DeleteTrackerFile
from gslib.tracker_file import GetRewriteTrackerFilePath
from gslib.tracker_file import GetSlicedDownloadTrackerFilePaths
from gslib.ui_controller import BytesToFixedWidthString
from gslib.utils import hashing_helper
from gslib.utils.boto_util import UsingCrcmodExtension
from gslib.utils.constants import START_CALLBACK_PER_BYTES
from gslib.utils.constants import UTF8
from gslib.utils.copy_helper import GetTrackerFilePath
from gslib.utils.copy_helper import PARALLEL_UPLOAD_STATIC_SALT
from gslib.utils.copy_helper import PARALLEL_UPLOAD_TEMP_NAMESPACE
from gslib.utils.copy_helper import TrackerFileType
from gslib.utils.hashing_helper import CalculateB64EncodedMd5FromContents
from gslib.utils.hashing_helper import CalculateMd5FromContents
from gslib.utils.hashing_helper import GetMd5
from gslib.utils.metadata_util import CreateCustomMetadata
from gslib.utils.posix_util import GID_ATTR
from gslib.utils.posix_util import MODE_ATTR
from gslib.utils.posix_util import NA_ID
from gslib.utils.posix_util import NA_MODE
from gslib.utils.posix_util import UID_ATTR
from gslib.utils.posix_util import ParseAndSetPOSIXAttributes
from gslib.utils.posix_util import ValidateFilePermissionAccess
from gslib.utils.posix_util import ValidatePOSIXMode
from gslib.utils.retry_util import Retry
from gslib.utils.system_util import IS_WINDOWS
from gslib.utils.text_util import get_random_ascii_chars
from gslib.utils.unit_util import EIGHT_MIB
from gslib.utils.unit_util import HumanReadableToBytes
from gslib.utils.unit_util import MakeHumanReadable
from gslib.utils.unit_util import ONE_KIB
from gslib.utils.unit_util import ONE_MIB
from gslib.utils import shim_util
import six
from six.moves import http_client
from six.moves import range
from six.moves import xrange
class TestCp(testcase.GsUtilIntegrationTestCase):
    """Integration tests for cp command."""
    halt_size = START_CALLBACK_PER_BYTES * 2

    def _get_test_file(self, name):
        contents = pkgutil.get_data('gslib', 'tests/test_data/%s' % name)
        return self.CreateTempFile(file_name=name, contents=contents)

    def _CpWithFifoViaGsUtilAndAppendOutputToList(self, src_path_tuple, dst_path, list_for_return_value, **kwargs):
        arg_list = ['cp']
        arg_list.extend(src_path_tuple)
        arg_list.append(dst_path)
        list_for_return_value.append(self.RunGsUtil(arg_list, **kwargs))

    @SequentialAndParallelTransfer
    def test_noclobber(self):
        key_uri = self.CreateObject(contents=b'foo')
        fpath = self.CreateTempFile(contents=b'bar')
        stderr = self.RunGsUtil(['cp', '-n', fpath, suri(key_uri)], return_stderr=True)
        self.assertRegex(stderr, 'Skipping.*: {}'.format(re.escape(suri(key_uri))))
        self.assertEqual(key_uri.get_contents_as_string(), b'foo')
        stderr = self.RunGsUtil(['cp', '-n', suri(key_uri), fpath], return_stderr=True)
        with open(fpath, 'rb') as f:
            self.assertRegex(stderr, 'Skipping.*: {}'.format(re.escape(suri(f))))
            self.assertEqual(f.read(), b'bar')

    @SequentialAndParallelTransfer
    def test_noclobber_different_size(self):
        key_uri = self.CreateObject(contents=b'foo')
        fpath = self.CreateTempFile(contents=b'quux')
        stderr = self.RunGsUtil(['cp', '-n', fpath, suri(key_uri)], return_stderr=True)
        self.assertRegex(stderr, 'Skipping.*: {}'.format(re.escape(suri(key_uri))))
        self.assertEqual(key_uri.get_contents_as_string(), b'foo')
        stderr = self.RunGsUtil(['cp', '-n', suri(key_uri), fpath], return_stderr=True)
        with open(fpath, 'rb') as f:
            self.assertRegex(stderr, 'Skipping.*: {}'.format(re.escape(suri(f))))
            self.assertEqual(f.read(), b'quux')

    def test_dest_bucket_not_exist(self):
        fpath = self.CreateTempFile(contents=b'foo')
        invalid_bucket_uri = '%s://%s' % (self.default_provider, self.nonexistent_bucket_name)

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check():
            stderr = self.RunGsUtil(['cp', fpath, invalid_bucket_uri], expected_status=1, return_stderr=True)
            if self._use_gcloud_storage:
                self.assertIn('not found: 404', stderr)
            else:
                self.assertIn('does not exist', stderr)
        _Check()

    def test_copy_in_cloud_noclobber(self):
        bucket1_uri = self.CreateBucket()
        bucket2_uri = self.CreateBucket()
        key_uri = self.CreateObject(bucket_uri=bucket1_uri, contents=b'foo')
        stderr = self.RunGsUtil(['cp', suri(key_uri), suri(bucket2_uri)], return_stderr=True)
        self.assertGreaterEqual(stderr.count('Copying'), 1)
        self.assertLessEqual(stderr.count('Copying'), 2)
        stderr = self.RunGsUtil(['cp', '-n', suri(key_uri), suri(bucket2_uri)], return_stderr=True)
        self.assertRegex(stderr, 'Skipping.*: {}'.format(suri(bucket2_uri, key_uri.object_name)))

    @SequentialAndParallelTransfer
    @SkipForXML('Boto library does not handle objects with .. in them.')
    def test_skip_object_with_parent_directory_symbol_in_name(self):
        bucket_uri = self.CreateBucket()
        key_uri = self.CreateObject(bucket_uri=bucket_uri, object_name='dir/../../../file', contents=b'data', prefer_json_api=True)
        self.CreateObject(bucket_uri=bucket_uri, object_name='file2', contents=b'data')
        directory = self.CreateTempDir()
        stderr = self.RunGsUtil(['cp', '-r', suri(bucket_uri), directory], return_stderr=True)
        self.json_api.DeleteObject(bucket_uri.bucket_name, key_uri.object_name)
        self.assertIn('Skipping copy of source URL %s because it would be copied outside the expected destination directory: %s.' % (suri(key_uri), os.path.abspath(directory)), stderr)
        self.assertFalse(os.path.exists(os.path.join(directory, 'file')))
        self.assertTrue(os.path.exists(os.path.join(directory, bucket_uri.bucket_name, 'file2')))

    @SequentialAndParallelTransfer
    @SkipForXML('Boto library does not handle objects with .. in them.')
    def test_skip_parent_directory_symbol_in_name_is_reflected_in_manifest(self):
        bucket_uri = self.CreateBucket()
        key_uri = self.CreateObject(bucket_uri=bucket_uri, object_name='dir/../../../file', contents=b'data', prefer_json_api=True)
        directory = self.CreateTempDir()
        log_path = os.path.join(directory, 'log.csv')
        stderr = self.RunGsUtil(['cp', '-r', '-L', log_path, suri(bucket_uri), directory], return_stderr=True)
        self.json_api.DeleteObject(bucket_uri.bucket_name, key_uri.object_name)
        self.assertIn('Skipping copy of source URL %s because it would be copied outside the expected destination directory: %s.' % (suri(key_uri), os.path.abspath(directory)), stderr)
        self.assertFalse(os.path.exists(os.path.join(directory, 'file')))
        with open(log_path, 'r') as f:
            lines = f.readlines()
            results = lines[1].strip().split(',')
            self.assertEqual(results[0], suri(key_uri))
            self.assertEqual(results[8], 'skip')

    @SequentialAndParallelTransfer
    @SkipForXML('Boto library does not handle objects with .. in them.')
    @unittest.skipIf(IS_WINDOWS, 'os.symlink() is not available on Windows.')
    def test_skip_parent_directory_symbol_object_with_symlink_destination(self):
        bucket_uri = self.CreateBucket()
        key_uri = self.CreateObject(bucket_uri=bucket_uri, object_name='dir/../../../file', contents=b'data', prefer_json_api=True)
        second_key_uri = self.CreateObject(bucket_uri=bucket_uri, object_name='file2', contents=b'data')
        directory = self.CreateTempDir()
        linked_destination = os.path.join(directory, 'linked_destination')
        destination = os.path.join(directory, 'destination')
        os.mkdir(destination)
        os.symlink(destination, linked_destination)
        stderr = self.RunGsUtil(['-D', 'cp', '-r', suri(bucket_uri), suri(second_key_uri), linked_destination], return_stderr=True)
        self.json_api.DeleteObject(bucket_uri.bucket_name, key_uri.object_name)
        self.assertIn('Skipping copy of source URL %s because it would be copied outside the expected destination directory: %s.' % (suri(key_uri), linked_destination), stderr)
        self.assertFalse(os.path.exists(os.path.join(linked_destination, 'file')))
        self.assertTrue(os.path.exists(os.path.join(linked_destination, 'file2')))

    @unittest.skipIf(IS_WINDOWS, 'os.mkfifo not available on Windows.')
    @SequentialAndParallelTransfer
    def test_cp_from_local_file_to_fifo(self):
        contents = b'bar'
        fifo_path = self.CreateTempFifo()
        file_path = self.CreateTempFile(contents=contents)
        list_for_output = []
        read_thread = threading.Thread(target=_ReadContentsFromFifo, args=(fifo_path, list_for_output))
        read_thread.start()
        write_thread = threading.Thread(target=self._CpWithFifoViaGsUtilAndAppendOutputToList, args=((file_path,), fifo_path, []))
        write_thread.start()
        write_thread.join(120)
        read_thread.join(120)
        if not list_for_output:
            self.fail('Reading/writing to the fifo timed out.')
        self.assertEqual(list_for_output[0].strip(), contents)

    @unittest.skipIf(IS_WINDOWS, 'os.mkfifo not available on Windows.')
    @SequentialAndParallelTransfer
    def test_cp_from_one_object_to_fifo(self):
        fifo_path = self.CreateTempFifo()
        bucket_uri = self.CreateBucket()
        contents = b'bar'
        obj_uri = self.CreateObject(bucket_uri=bucket_uri, contents=contents)
        list_for_output = []
        read_thread = threading.Thread(target=_ReadContentsFromFifo, args=(fifo_path, list_for_output))
        read_thread.start()
        write_thread = threading.Thread(target=self._CpWithFifoViaGsUtilAndAppendOutputToList, args=((suri(obj_uri),), fifo_path, []))
        write_thread.start()
        write_thread.join(120)
        read_thread.join(120)
        if not list_for_output:
            self.fail('Reading/writing to the fifo timed out.')
        self.assertEqual(list_for_output[0].strip(), contents)

    @unittest.skipIf(IS_WINDOWS, 'os.mkfifo not available on Windows.')
    @SequentialAndParallelTransfer
    def test_cp_from_multiple_objects_to_fifo(self):
        fifo_path = self.CreateTempFifo()
        bucket_uri = self.CreateBucket()
        contents1 = b'foo and bar'
        contents2 = b'baz and qux'
        obj1_uri = self.CreateObject(bucket_uri=bucket_uri, contents=contents1)
        obj2_uri = self.CreateObject(bucket_uri=bucket_uri, contents=contents2)
        list_for_output = []
        read_thread = threading.Thread(target=_ReadContentsFromFifo, args=(fifo_path, list_for_output))
        read_thread.start()
        write_thread = threading.Thread(target=self._CpWithFifoViaGsUtilAndAppendOutputToList, args=((suri(obj1_uri), suri(obj2_uri)), fifo_path, []))
        write_thread.start()
        write_thread.join(120)
        read_thread.join(120)
        if not list_for_output:
            self.fail('Reading/writing to the fifo timed out.')
        self.assertIn(contents1, list_for_output[0])
        self.assertIn(contents2, list_for_output[0])

    @SequentialAndParallelTransfer
    def test_streaming(self):
        bucket_uri = self.CreateBucket()
        stderr = self.RunGsUtil(['cp', '-', '%s' % suri(bucket_uri, 'foo')], stdin='bar', return_stderr=True)
        if self._use_gcloud_storage:
            self.assertIn('Copying file://- to ' + suri(bucket_uri, 'foo'), stderr)
        else:
            self.assertIn('Copying from <STDIN>', stderr)
        key_uri = self.StorageUriCloneReplaceName(bucket_uri, 'foo')
        self.assertEqual(key_uri.get_contents_as_string(), b'bar')

    @unittest.skipIf(IS_WINDOWS, 'os.mkfifo not available on Windows.')
    @SequentialAndParallelTransfer
    def test_streaming_from_fifo_to_object(self):
        bucket_uri = self.CreateBucket()
        fifo_path = self.CreateTempFifo()
        object_name = 'foo'
        object_contents = b'bar'
        list_for_output = []
        write_thread = threading.Thread(target=_WriteContentsToFifo, args=(object_contents, fifo_path))
        write_thread.start()
        read_thread = threading.Thread(target=self._CpWithFifoViaGsUtilAndAppendOutputToList, args=((fifo_path,), suri(bucket_uri, object_name), list_for_output), kwargs={'return_stderr': True})
        read_thread.start()
        read_thread.join(120)
        write_thread.join(120)
        if not list_for_output:
            self.fail('Reading/writing to the fifo timed out.')
        if self._use_gcloud_storage:
            self.assertIn('Copying file://{} to {}'.format(fifo_path, suri(bucket_uri, object_name)), list_for_output[0])
        else:
            self.assertIn('Copying from named pipe', list_for_output[0])
        key_uri = self.StorageUriCloneReplaceName(bucket_uri, object_name)
        self.assertEqual(key_uri.get_contents_as_string(), object_contents)

    @unittest.skipIf(IS_WINDOWS, 'os.mkfifo not available on Windows.')
    @SequentialAndParallelTransfer
    def test_streaming_from_fifo_to_stdout(self):
        fifo_path = self.CreateTempFifo()
        contents = b'bar'
        list_for_output = []
        write_thread = threading.Thread(target=_WriteContentsToFifo, args=(contents, fifo_path))
        write_thread.start()
        read_thread = threading.Thread(target=self._CpWithFifoViaGsUtilAndAppendOutputToList, args=((fifo_path,), '-', list_for_output), kwargs={'return_stdout': True})
        read_thread.start()
        read_thread.join(120)
        write_thread.join(120)
        if not list_for_output:
            self.fail('Reading/writing to the fifo timed out.')
        self.assertEqual(list_for_output[0].strip().encode('ascii'), contents)

    @unittest.skipIf(IS_WINDOWS, 'os.mkfifo not available on Windows.')
    @SequentialAndParallelTransfer
    def test_streaming_from_stdout_to_fifo(self):
        fifo_path = self.CreateTempFifo()
        contents = b'bar'
        list_for_output = []
        list_for_gsutil_output = []
        read_thread = threading.Thread(target=_ReadContentsFromFifo, args=(fifo_path, list_for_output))
        read_thread.start()
        write_thread = threading.Thread(target=self._CpWithFifoViaGsUtilAndAppendOutputToList, args=(('-',), fifo_path, list_for_gsutil_output), kwargs={'return_stderr': True, 'stdin': contents})
        write_thread.start()
        write_thread.join(120)
        read_thread.join(120)
        if not list_for_output:
            self.fail('Reading/writing to the fifo timed out.')
        self.assertEqual(list_for_output[0].strip(), contents)

    def test_streaming_multiple_arguments(self):
        bucket_uri = self.CreateBucket()
        stderr = self.RunGsUtil(['cp', '-', '-', suri(bucket_uri)], stdin='bar', return_stderr=True, expected_status=1)
        if self._use_gcloud_storage:
            self.assertIn('Multiple URL strings are not supported when transferring from stdin.', stderr)
        else:
            self.assertIn('Multiple URL strings are not supported with streaming', stderr)

    @SequentialAndParallelTransfer
    def test_detect_content_type(self):
        """Tests local detection of content type."""
        bucket_uri = self.CreateBucket()
        dsturi = suri(bucket_uri, 'foo')
        self.RunGsUtil(['cp', self._get_test_file('test.mp3'), dsturi])

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check1():
            stdout = self.RunGsUtil(['ls', '-L', dsturi], return_stdout=True)
            if IS_WINDOWS:
                self.assertTrue(re.search('Content-Type:\\s+audio/x-mpg', stdout) or re.search('Content-Type:\\s+audio/mpeg', stdout))
            else:
                self.assertRegex(stdout, 'Content-Type:\\s+audio/mpeg')
        _Check1()
        self.RunGsUtil(['cp', self._get_test_file('test.gif'), dsturi])

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check2():
            stdout = self.RunGsUtil(['ls', '-L', dsturi], return_stdout=True)
            self.assertRegex(stdout, 'Content-Type:\\s+image/gif')
        _Check2()

    def test_content_type_override_default(self):
        """Tests overriding content type with the default value."""
        bucket_uri = self.CreateBucket()
        dsturi = suri(bucket_uri, 'foo')
        self.RunGsUtil(['-h', 'Content-Type:', 'cp', self._get_test_file('test.mp3'), dsturi])

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check1():
            stdout = self.RunGsUtil(['ls', '-L', dsturi], return_stdout=True)
            self.assertRegex(stdout, 'Content-Type:\\s+application/octet-stream')
        _Check1()
        self.RunGsUtil(['-h', 'Content-Type:', 'cp', self._get_test_file('test.gif'), dsturi])

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check2():
            stdout = self.RunGsUtil(['ls', '-L', dsturi], return_stdout=True)
            self.assertRegex(stdout, 'Content-Type:\\s+application/octet-stream')
        _Check2()

    def test_content_type_override(self):
        """Tests overriding content type with a value."""
        bucket_uri = self.CreateBucket()
        dsturi = suri(bucket_uri, 'foo')
        self.RunGsUtil(['-h', 'Content-Type:text/plain', 'cp', self._get_test_file('test.mp3'), dsturi])

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check1():
            stdout = self.RunGsUtil(['ls', '-L', dsturi], return_stdout=True)
            self.assertRegex(stdout, 'Content-Type:\\s+text/plain')
        _Check1()
        self.RunGsUtil(['-h', 'Content-Type:text/plain', 'cp', self._get_test_file('test.gif'), dsturi])

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check2():
            stdout = self.RunGsUtil(['ls', '-L', dsturi], return_stdout=True)
            self.assertRegex(stdout, 'Content-Type:\\s+text/plain')
        _Check2()

    @unittest.skipIf(IS_WINDOWS, 'magicfile is not available on Windows.')
    @SequentialAndParallelTransfer
    def test_magicfile_override(self):
        """Tests content type override with magicfile value."""
        bucket_uri = self.CreateBucket()
        dsturi = suri(bucket_uri, 'foo')
        fpath = self.CreateTempFile(contents=b'foo/bar\n')
        self.RunGsUtil(['cp', fpath, dsturi])

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check1():
            stdout = self.RunGsUtil(['ls', '-L', dsturi], return_stdout=True)
            use_magicfile = boto.config.getbool('GSUtil', 'use_magicfile', False)
            content_type = 'text/plain' if use_magicfile else 'application/octet-stream'
            self.assertRegex(stdout, 'Content-Type:\\s+%s' % content_type)
        _Check1()

    @SequentialAndParallelTransfer
    def test_content_type_mismatches(self):
        """Tests overriding content type when it does not match the file type."""
        bucket_uri = self.CreateBucket()
        dsturi = suri(bucket_uri, 'foo')
        fpath = self.CreateTempFile(contents=b'foo/bar\n')
        self.RunGsUtil(['-h', 'Content-Type:image/gif', 'cp', self._get_test_file('test.mp3'), dsturi])

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check1():
            stdout = self.RunGsUtil(['ls', '-L', dsturi], return_stdout=True)
            self.assertRegex(stdout, 'Content-Type:\\s+image/gif')
        _Check1()
        self.RunGsUtil(['-h', 'Content-Type:image/gif', 'cp', self._get_test_file('test.gif'), dsturi])

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check2():
            stdout = self.RunGsUtil(['ls', '-L', dsturi], return_stdout=True)
            self.assertRegex(stdout, 'Content-Type:\\s+image/gif')
        _Check2()
        self.RunGsUtil(['-h', 'Content-Type:image/gif', 'cp', fpath, dsturi])

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check3():
            stdout = self.RunGsUtil(['ls', '-L', dsturi], return_stdout=True)
            self.assertRegex(stdout, 'Content-Type:\\s+image/gif')
        _Check3()

    @SequentialAndParallelTransfer
    def test_content_type_header_case_insensitive(self):
        """Tests that content type header is treated with case insensitivity."""
        bucket_uri = self.CreateBucket()
        dsturi = suri(bucket_uri, 'foo')
        fpath = self._get_test_file('test.gif')
        self.RunGsUtil(['-h', 'content-Type:text/plain', 'cp', fpath, dsturi])

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check1():
            stdout = self.RunGsUtil(['ls', '-L', dsturi], return_stdout=True)
            self.assertRegex(stdout, 'Content-Type:\\s+text/plain')
            self.assertNotRegex(stdout, 'image/gif')
        _Check1()
        self.RunGsUtil(['-h', 'CONTENT-TYPE:image/gif', '-h', 'content-type:image/gif', 'cp', fpath, dsturi])

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check2():
            stdout = self.RunGsUtil(['ls', '-L', dsturi], return_stdout=True)
            self.assertRegex(stdout, 'Content-Type:\\s+image/gif')
            self.assertNotRegex(stdout, 'image/gif,\\s*image/gif')
        _Check2()

    @SequentialAndParallelTransfer
    def test_other_headers(self):
        """Tests that non-content-type headers are applied successfully on copy."""
        bucket_uri = self.CreateBucket()
        dst_uri = suri(bucket_uri, 'foo')
        fpath = self._get_test_file('test.gif')
        self.RunGsUtil(['-h', 'Cache-Control:public,max-age=12', '-h', 'x-%s-meta-1:abcd' % self.provider_custom_meta, 'cp', fpath, dst_uri])
        stdout = self.RunGsUtil(['ls', '-L', dst_uri], return_stdout=True)
        self.assertRegex(stdout, 'Cache-Control\\s*:\\s*public,max-age=12')
        self.assertRegex(stdout, 'Metadata:\\s*1:\\s*abcd')
        dst_uri2 = suri(bucket_uri, 'bar')
        self.RunGsUtil(['cp', dst_uri, dst_uri2])
        stdout = self.RunGsUtil(['ls', '-L', dst_uri2], return_stdout=True)
        self.assertRegex(stdout, 'Cache-Control\\s*:\\s*public,max-age=12')
        self.assertRegex(stdout, 'Metadata:\\s*1:\\s*abcd')

    @SequentialAndParallelTransfer
    def test_request_reason_header(self):
        """Test that x-goog-request-header can be set using the environment variable."""
        os.environ['CLOUDSDK_CORE_REQUEST_REASON'] = 'b/this_is_env_reason'
        bucket_uri = self.CreateBucket()
        dst_uri = suri(bucket_uri, 'foo')
        fpath = self._get_test_file('test.gif')
        stderr = self.RunGsUtil(['-DD', 'cp', fpath, dst_uri], return_stderr=True)
        if self._use_gcloud_storage:
            reason_regex = "b'X-Goog-Request-Reason': b'b/this_is_env_reason'"
        else:
            reason_regex = "'x-goog-request-reason': 'b/this_is_env_reason'"
        self.assertRegex(stderr, reason_regex)
        stderr = self.RunGsUtil(['-DD', 'ls', '-L', dst_uri], return_stderr=True)
        self.assertRegex(stderr, reason_regex)

    @SequentialAndParallelTransfer
    @SkipForXML('XML APIs use a different debug log format.')
    def test_request_reason_header_persists_multiple_requests_json(self):
        """Test that x-goog-request-header works when cp sends multiple requests."""
        os.environ['CLOUDSDK_CORE_REQUEST_REASON'] = 'b/this_is_env_reason'
        bucket_uri = self.CreateBucket()
        dst_uri = suri(bucket_uri, 'foo')
        fpath = self._get_test_file('test.gif')
        boto_config_for_test = ('GSUtil', 'resumable_threshold', '0')
        with SetBotoConfigForTest([boto_config_for_test]):
            stderr = self.RunGsUtil(['-DD', 'cp', fpath, dst_uri], return_stderr=True)
        if self._use_gcloud_storage:
            reason_regex = "X-Goog-Request-Reason\\': b\\'b/this_is_env_reason"
        else:
            reason_regex = "x-goog-request-reason\\': \\'b/this_is_env_reason"
        self.assertRegex(stderr, 'GET[\\s\\S]*' + reason_regex + '[\\s\\S]*POST[\\s\\S]*' + reason_regex)

    @SequentialAndParallelTransfer
    @SkipForJSON('JSON API uses a different debug log format.')
    def test_request_reason_header_persists_multiple_requests_xml(self):
        """Test that x-goog-request-header works when cp sends multiple requests."""
        os.environ['CLOUDSDK_CORE_REQUEST_REASON'] = 'b/this_is_env_reason'
        bucket_uri = self.CreateBucket()
        dst_uri = suri(bucket_uri, 'foo')
        fpath = self._get_test_file('test.gif')
        boto_config_for_test = ('GSUtil', 'resumable_threshold', '0')
        with SetBotoConfigForTest([boto_config_for_test]):
            stderr = self.RunGsUtil(['-D', 'cp', fpath, dst_uri], return_stderr=True)
        reason_regex = "Final headers: \\{[\\s\\S]*\\'x-goog-request-reason\\': \\'b/this_is_env_reason\\'[\\s\\S]*}"
        self.assertRegex(stderr, reason_regex + '[\\s\\S]*' + reason_regex)

    @SequentialAndParallelTransfer
    def test_versioning(self):
        """Tests copy with versioning."""
        bucket_uri = self.CreateVersionedBucket()
        k1_uri = self.CreateObject(bucket_uri=bucket_uri, contents=b'data2')
        k2_uri = self.CreateObject(bucket_uri=bucket_uri, contents=b'data1')
        g1 = urigen(k2_uri)
        self.RunGsUtil(['cp', suri(k1_uri), suri(k2_uri)])
        k2_uri = self.StorageUriCloneReplaceName(bucket_uri, k2_uri.object_name)
        k2_uri = self.StorageUriCloneReplaceKey(bucket_uri, k2_uri.get_key())
        g2 = urigen(k2_uri)
        self.StorageUriSetContentsFromString(k2_uri, 'data3')
        g3 = urigen(k2_uri)
        fpath = self.CreateTempFile()
        self.RunGsUtil(['cp', k2_uri.versionless_uri, fpath])
        with open(fpath, 'rb') as f:
            self.assertEqual(f.read(), b'data3')
        self.RunGsUtil(['cp', '%s#%s' % (k2_uri.versionless_uri, g1), fpath])
        with open(fpath, 'rb') as f:
            self.assertEqual(f.read(), b'data1')
        self.RunGsUtil(['cp', '%s#%s' % (k2_uri.versionless_uri, g2), fpath])
        with open(fpath, 'rb') as f:
            self.assertEqual(f.read(), b'data2')
        self.RunGsUtil(['cp', '%s#%s' % (k2_uri.versionless_uri, g3), fpath])
        with open(fpath, 'rb') as f:
            self.assertEqual(f.read(), b'data3')
        self.RunGsUtil(['cp', '%s#%s' % (k2_uri.versionless_uri, g1), k2_uri.versionless_uri])
        self.RunGsUtil(['cp', k2_uri.versionless_uri, fpath])
        with open(fpath, 'rb') as f:
            self.assertEqual(f.read(), b'data1')
        stderr = self.RunGsUtil(['cp', fpath, k2_uri.uri], return_stderr=True, expected_status=1)
        if self._use_gcloud_storage:
            self.assertIn('destination argument of the cp command cannot be a version-specific URL', stderr)
        else:
            self.assertIn('cannot be the destination for gsutil cp', stderr)

    def test_versioning_no_parallelism(self):
        """Tests that copy all-versions errors when parallelism is enabled."""

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check():
            stderr = self.RunGsUtil(['-m', 'cp', '-A', suri(self.nonexistent_bucket_name, 'foo'), suri(self.nonexistent_bucket_name, 'bar')], expected_status=1, return_stderr=True)
            if self._use_gcloud_storage:
                self.assertIn('sequential instead of parallel task execution', stderr)
            else:
                self.assertIn('-m option is not supported with the cp -A flag', stderr)
        _Check()

    @SkipForS3('S3 lists versioned objects in reverse timestamp order.')
    def test_recursive_copying_versioned_bucket(self):
        """Tests cp -R with versioned buckets."""
        bucket1_uri = self.CreateVersionedBucket()
        bucket2_uri = self.CreateVersionedBucket()
        bucket3_uri = self.CreateVersionedBucket()
        v1_uri = self.CreateObject(bucket_uri=bucket1_uri, object_name='k', contents=b'data0')
        self.CreateObject(bucket_uri=bucket1_uri, object_name='k', contents=b'longer_data1', gs_idempotent_generation=urigen(v1_uri))
        self.AssertNObjectsInBucket(bucket1_uri, 2, versioned=True)
        self.AssertNObjectsInBucket(bucket2_uri, 0, versioned=True)
        self.AssertNObjectsInBucket(bucket3_uri, 0, versioned=True)
        self.RunGsUtil(['cp', '-R', '-A', suri(bucket1_uri, '*'), suri(bucket2_uri)])

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check2():
            """Validates the results of the cp -R."""
            listing1 = self.RunGsUtil(['ls', '-la', suri(bucket1_uri)], return_stdout=True).split('\n')
            listing2 = self.RunGsUtil(['ls', '-la', suri(bucket2_uri)], return_stdout=True).split('\n')
            self.assertEqual(len(listing1), 4)
            self.assertEqual(len(listing2), 4)
            size1, _, uri_str1, _ = listing1[0].split()
            self.assertEqual(size1, str(len('data0')))
            self.assertEqual(storage_uri(uri_str1).object_name, 'k')
            size2, _, uri_str2, _ = listing2[0].split()
            self.assertEqual(size2, str(len('data0')))
            self.assertEqual(storage_uri(uri_str2).object_name, 'k')
            size1, _, uri_str1, _ = listing1[1].split()
            self.assertEqual(size1, str(len('longer_data1')))
            self.assertEqual(storage_uri(uri_str1).object_name, 'k')
            size2, _, uri_str2, _ = listing2[1].split()
            self.assertEqual(size2, str(len('longer_data1')))
            self.assertEqual(storage_uri(uri_str2).object_name, 'k')
        _Check2()
        self.RunGsUtil(['cp', '-R', suri(bucket1_uri, '*'), suri(bucket3_uri)])

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check3():
            """Validates the results of the cp -R."""
            listing1 = self.RunGsUtil(['ls', '-la', suri(bucket1_uri)], return_stdout=True).split('\n')
            listing2 = self.RunGsUtil(['ls', '-la', suri(bucket3_uri)], return_stdout=True).split('\n')
            self.assertEqual(len(listing1), 4)
            self.assertEqual(len(listing2), 3)
            size1, _, uri_str1, _ = listing2[0].split()
            self.assertEqual(size1, str(len('longer_data1')))
            self.assertEqual(storage_uri(uri_str1).object_name, 'k')
        _Check3()

    @SequentialAndParallelTransfer
    @SkipForS3('Preconditions not supported for S3.')
    def test_cp_generation_zero_match(self):
        """Tests that cp handles an object-not-exists precondition header."""
        bucket_uri = self.CreateBucket()
        fpath1 = self.CreateTempFile(contents=b'data1')
        gen_match_header = 'x-goog-if-generation-match:0'
        self.RunGsUtil(['-h', gen_match_header, 'cp', fpath1, suri(bucket_uri)])
        stderr = self.RunGsUtil(['-h', gen_match_header, 'cp', fpath1, suri(bucket_uri)], return_stderr=True, expected_status=1)
        if self._use_gcloud_storage:
            self.assertIn('HTTPError 412: At least one of the pre-conditions you specified did not hold.', stderr)
        else:
            self.assertIn('PreconditionException', stderr)

    @SequentialAndParallelTransfer
    @SkipForS3('Preconditions not supported for S3.')
    def test_cp_v_generation_match(self):
        """Tests that cp -v option handles the if-generation-match header."""
        bucket_uri = self.CreateVersionedBucket()
        k1_uri = self.CreateObject(bucket_uri=bucket_uri, contents=b'data1')
        g1 = k1_uri.generation
        tmpdir = self.CreateTempDir()
        fpath1 = self.CreateTempFile(tmpdir=tmpdir, contents=b'data2')
        gen_match_header = 'x-goog-if-generation-match:%s' % g1
        self.RunGsUtil(['-h', gen_match_header, 'cp', fpath1, suri(k1_uri)])
        stderr = self.RunGsUtil(['-h', gen_match_header, 'cp', fpath1, suri(k1_uri)], return_stderr=True, expected_status=1)
        if self._use_gcloud_storage:
            self.assertIn('pre-condition', stderr)
        else:
            self.assertIn('PreconditionException', stderr)
        stderr = self.RunGsUtil(['-h', gen_match_header, 'cp', '-n', fpath1, suri(k1_uri)], return_stderr=True, expected_status=1)
        if self._use_gcloud_storage:
            self.assertIn('Cannot specify both generation precondition and no-clobber', stderr)
        else:
            self.assertIn('ArgumentException', stderr)
            self.assertIn('Specifying x-goog-if-generation-match is not supported with cp -n', stderr)

    @SequentialAndParallelTransfer
    def test_cp_nv(self):
        """Tests that cp -nv works when skipping existing file."""
        bucket_uri = self.CreateVersionedBucket()
        k1_uri = self.CreateObject(bucket_uri=bucket_uri, contents=b'data1')
        tmpdir = self.CreateTempDir()
        fpath1 = self.CreateTempFile(tmpdir=tmpdir, contents=b'data2')
        self.RunGsUtil(['cp', '-nv', fpath1, suri(k1_uri)])
        stderr = self.RunGsUtil(['cp', '-nv', fpath1, suri(k1_uri)], return_stderr=True)
        self.assertIn('Skipping existing', stderr)

    @SequentialAndParallelTransfer
    @SkipForS3('S3 lists versioned objects in reverse timestamp order.')
    def test_cp_v_option(self):
        """"Tests that cp -v returns the created object's version-specific URI."""
        bucket_uri = self.CreateVersionedBucket()
        k1_uri = self.CreateObject(bucket_uri=bucket_uri, contents=b'data1')
        k2_uri = self.CreateObject(bucket_uri=bucket_uri, contents=b'data2')
        tmpdir = self.CreateTempDir()
        fpath1 = self.CreateTempFile(tmpdir=tmpdir, contents=b'data1')
        self._run_cp_minus_v_test('-v', fpath1, k2_uri.uri)
        size_threshold = ONE_KIB
        boto_config_for_test = ('GSUtil', 'resumable_threshold', str(size_threshold))
        with SetBotoConfigForTest([boto_config_for_test]):
            file_as_string = os.urandom(size_threshold)
            tmpdir = self.CreateTempDir()
            fpath1 = self.CreateTempFile(tmpdir=tmpdir, contents=file_as_string)
            self._run_cp_minus_v_test('-v', fpath1, k2_uri.uri)
        self._run_cp_minus_v_test('-v', '-', k2_uri.uri)
        tmpdir = self.CreateTempDir()
        fpath1 = self.CreateTempFile(tmpdir=tmpdir)
        dst_uri = storage_uri(fpath1)
        stderr = self.RunGsUtil(['cp', '-v', suri(k1_uri), suri(dst_uri)], return_stderr=True)
        self.assertIn('Created: %s\n' % dst_uri.uri, stderr)
        self._run_cp_minus_v_test('-Dv', k1_uri.uri, k2_uri.uri)
        self._run_cp_minus_v_test('-v', k1_uri.uri, k2_uri.uri)

    def _run_cp_minus_v_test(self, opt, src_str, dst_str):
        """Runs cp -v with the options and validates the results."""
        stderr = self.RunGsUtil(['cp', opt, src_str, dst_str], return_stderr=True)
        match = re.search('Created: (.*)\\n', stderr)
        self.assertIsNotNone(match)
        created_uri = match.group(1)

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check1():
            stdout = self.RunGsUtil(['ls', '-a', dst_str], return_stdout=True)
            lines = stdout.split('\n')
            self.assertGreater(len(lines), 2)
            self.assertEqual(created_uri, lines[-2])
        _Check1()

    @SequentialAndParallelTransfer
    def test_stdin_args(self):
        """Tests cp with the -I option."""
        tmpdir = self.CreateTempDir()
        fpath1 = self.CreateTempFile(tmpdir=tmpdir, contents=b'data1')
        fpath2 = self.CreateTempFile(tmpdir=tmpdir, contents=b'data2')
        bucket_uri = self.CreateBucket()
        self.RunGsUtil(['cp', '-I', suri(bucket_uri)], stdin='\n'.join((fpath1, fpath2)))

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check1():
            stdout = self.RunGsUtil(['ls', suri(bucket_uri)], return_stdout=True)
            self.assertIn(os.path.basename(fpath1), stdout)
            self.assertIn(os.path.basename(fpath2), stdout)
            self.assertNumLines(stdout, 2)
        _Check1()

    def test_cross_storage_class_cloud_cp(self):
        bucket1_uri = self.CreateBucket(storage_class='standard')
        bucket2_uri = self.CreateBucket(storage_class='durable_reduced_availability')
        key_uri = self.CreateObject(bucket_uri=bucket1_uri, contents=b'foo')
        self.RunGsUtil(['cp', suri(key_uri), suri(bucket2_uri)])

    @unittest.skipUnless(HAS_S3_CREDS, 'Test requires both S3 and GS credentials')
    def test_cross_provider_cp(self):
        s3_bucket = self.CreateBucket(provider='s3')
        gs_bucket = self.CreateBucket(provider='gs')
        s3_key = self.CreateObject(bucket_uri=s3_bucket, contents=b'foo')
        gs_key = self.CreateObject(bucket_uri=gs_bucket, contents=b'bar')
        self.RunGsUtil(['cp', suri(s3_key), suri(gs_bucket)])
        self.RunGsUtil(['cp', suri(gs_key), suri(s3_bucket)])

    @unittest.skipUnless(HAS_S3_CREDS, 'Test requires both S3 and GS credentials')
    @unittest.skip('This test performs a large copy but remains here for debugging purposes.')
    def test_cross_provider_large_cp(self):
        s3_bucket = self.CreateBucket(provider='s3')
        gs_bucket = self.CreateBucket(provider='gs')
        s3_key = self.CreateObject(bucket_uri=s3_bucket, contents=b'f' * 1024 * 1024)
        gs_key = self.CreateObject(bucket_uri=gs_bucket, contents=b'b' * 1024 * 1024)
        self.RunGsUtil(['cp', suri(s3_key), suri(gs_bucket)])
        self.RunGsUtil(['cp', suri(gs_key), suri(s3_bucket)])
        with SetBotoConfigForTest([('GSUtil', 'resumable_threshold', str(ONE_KIB)), ('GSUtil', 'json_resumable_chunk_size', str(ONE_KIB * 256))]):
            self.RunGsUtil(['cp', suri(s3_key), suri(gs_bucket)])

    @unittest.skipUnless(HAS_S3_CREDS, 'Test requires both S3 and GS credentials')
    def test_gs_to_s3_multipart_cp(self):
        """Ensure daisy_chain works for an object that is downloaded in 2 parts."""
        s3_bucket = self.CreateBucket(provider='s3')
        gs_bucket = self.CreateBucket(provider='gs', prefer_json_api=True)
        num_bytes = int(_DEFAULT_DOWNLOAD_CHUNK_SIZE * 1.1)
        gs_key = self.CreateObject(bucket_uri=gs_bucket, contents=b'b' * num_bytes, prefer_json_api=True)
        self.RunGsUtil(['-o', 's3:use-sigv4=True', '-o', 's3:host=s3.amazonaws.com', 'cp', suri(gs_key), suri(s3_bucket)])

    @unittest.skip('This test is slow due to creating many objects, but remains here for debugging purposes.')
    def test_daisy_chain_cp_file_sizes(self):
        """Ensure daisy chain cp works with a wide of file sizes."""
        bucket_uri = self.CreateBucket()
        bucket2_uri = self.CreateBucket()
        exponent_cap = 28
        for i in range(exponent_cap):
            one_byte_smaller = 2 ** i - 1
            normal = 2 ** i
            one_byte_larger = 2 ** i + 1
            self.CreateObject(bucket_uri=bucket_uri, contents=b'a' * one_byte_smaller)
            self.CreateObject(bucket_uri=bucket_uri, contents=b'b' * normal)
            self.CreateObject(bucket_uri=bucket_uri, contents=b'c' * one_byte_larger)
        self.AssertNObjectsInBucket(bucket_uri, exponent_cap * 3)
        self.RunGsUtil(['-m', 'cp', '-D', suri(bucket_uri, '**'), suri(bucket2_uri)])
        self.AssertNObjectsInBucket(bucket2_uri, exponent_cap * 3)

    def test_daisy_chain_cp(self):
        """Tests cp with the -D option."""
        bucket1_uri = self.CreateBucket(storage_class='standard')
        bucket2_uri = self.CreateBucket(storage_class='durable_reduced_availability')
        key_uri = self.CreateObject(bucket_uri=bucket1_uri, contents=b'foo')
        self.RunGsUtil(['setmeta', '-h', 'Cache-Control:public,max-age=12', '-h', 'Content-Type:image/gif', '-h', 'x-%s-meta-1:abcd' % self.provider_custom_meta, suri(key_uri)])
        self.RunGsUtil(['acl', 'set', 'public-read', suri(key_uri)])
        acl_json = self.RunGsUtil(['acl', 'get', suri(key_uri)], return_stdout=True)
        stderr = self.RunGsUtil(['cp', '-Dpn', suri(key_uri), suri(bucket2_uri)], return_stderr=True)
        self.assertNotIn('Copy-in-the-cloud disallowed', stderr)

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check():
            uri = suri(bucket2_uri, key_uri.object_name)
            stdout = self.RunGsUtil(['ls', '-L', uri], return_stdout=True)
            self.assertRegex(stdout, 'Cache-Control:\\s+public,max-age=12')
            self.assertRegex(stdout, 'Content-Type:\\s+image/gif')
            self.assertRegex(stdout, 'Metadata:\\s+1:\\s+abcd')
            new_acl_json = self.RunGsUtil(['acl', 'get', uri], return_stdout=True)
            self.assertEqual(acl_json, new_acl_json)
        _Check()

    @unittest.skipUnless(not HAS_GS_PORT, 'gs_port is defined in config which can cause problems when uploading and downloading to the same local host port')
    def test_daisy_chain_cp_download_failure(self):
        """Tests cp with the -D option when the download thread dies."""
        bucket1_uri = self.CreateBucket()
        bucket2_uri = self.CreateBucket()
        key_uri = self.CreateObject(bucket_uri=bucket1_uri, contents=b'a' * self.halt_size)
        boto_config_for_test = ('GSUtil', 'resumable_threshold', str(ONE_KIB))
        test_callback_file = self.CreateTempFile(contents=pickle.dumps(HaltingCopyCallbackHandler(False, 5)))
        with SetBotoConfigForTest([boto_config_for_test]):
            stderr = self.RunGsUtil(['cp', '--testcallbackfile', test_callback_file, '-D', suri(key_uri), suri(bucket2_uri)], expected_status=1, return_stderr=True)
            self.assertEqual(stderr.count('ResumableDownloadException: Artifically halting download'), 3)

    def test_streaming_gzip_upload(self):
        """Tests error when compression flag is requested on a streaming source."""
        bucket_uri = self.CreateBucket()
        stderr = self.RunGsUtil(['cp', '-Z', '-', suri(bucket_uri, 'foo')], return_stderr=True, expected_status=1, stdin='streaming data')
        if self._use_gcloud_storage:
            self.assertIn('Gzip content encoding is not currently supported for streaming uploads.', stderr)
        else:
            self.assertIn('gzip compression is not currently supported on streaming uploads', stderr)

    def test_seek_ahead_upload_cp(self):
        """Tests that the seek-ahead iterator estimates total upload work."""
        tmpdir = self.CreateTempDir(test_files=3)
        bucket_uri = self.CreateBucket()
        with SetBotoConfigForTest([('GSUtil', 'task_estimation_threshold', '1'), ('GSUtil', 'task_estimation_force', 'True')]):
            stderr = self.RunGsUtil(['-m', 'cp', '-r', tmpdir, suri(bucket_uri)], return_stderr=True)
            self.assertIn('Estimated work for this command: objects: 3, total size: 18', stderr)
        with SetBotoConfigForTest([('GSUtil', 'task_estimation_threshold', '0'), ('GSUtil', 'task_estimation_force', 'True')]):
            stderr = self.RunGsUtil(['-m', 'cp', '-r', tmpdir, suri(bucket_uri)], return_stderr=True)
            self.assertNotIn('Estimated work', stderr)

    def test_seek_ahead_download_cp(self):
        tmpdir = self.CreateTempDir()
        bucket_uri = self.CreateBucket(test_objects=3)
        self.AssertNObjectsInBucket(bucket_uri, 3)
        with SetBotoConfigForTest([('GSUtil', 'task_estimation_threshold', '1'), ('GSUtil', 'task_estimation_force', 'True')]):
            stderr = self.RunGsUtil(['-m', 'cp', '-r', suri(bucket_uri), tmpdir], return_stderr=True)
            self.assertIn('Estimated work for this command: objects: 3, total size: 18', stderr)
        with SetBotoConfigForTest([('GSUtil', 'task_estimation_threshold', '0'), ('GSUtil', 'task_estimation_force', 'True')]):
            stderr = self.RunGsUtil(['-m', 'cp', '-r', suri(bucket_uri), tmpdir], return_stderr=True)
            self.assertNotIn('Estimated work', stderr)

    def test_canned_acl_cp(self):
        """Tests copying with a canned ACL."""
        bucket1_uri = self.CreateBucket()
        bucket2_uri = self.CreateBucket()
        key_uri = self.CreateObject(bucket_uri=bucket1_uri, contents=b'foo')
        self.RunGsUtil(['cp', '-a', 'public-read', suri(key_uri), suri(bucket2_uri)])
        self.RunGsUtil(['acl', 'set', 'public-read', suri(key_uri)])
        public_read_acl = self.RunGsUtil(['acl', 'get', suri(key_uri)], return_stdout=True)

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check():
            uri = suri(bucket2_uri, key_uri.object_name)
            new_acl_json = self.RunGsUtil(['acl', 'get', uri], return_stdout=True)
            self.assertEqual(public_read_acl, new_acl_json)
        _Check()

    @SequentialAndParallelTransfer
    def test_canned_acl_upload(self):
        """Tests uploading a file with a canned ACL."""
        bucket1_uri = self.CreateBucket()
        key_uri = self.CreateObject(bucket_uri=bucket1_uri, contents=b'foo')
        self.RunGsUtil(['acl', 'set', 'public-read', suri(key_uri)])
        public_read_acl = self.RunGsUtil(['acl', 'get', suri(key_uri)], return_stdout=True)
        file_name = 'bar'
        fpath = self.CreateTempFile(file_name=file_name, contents=b'foo')
        self.RunGsUtil(['cp', '-a', 'public-read', fpath, suri(bucket1_uri)])
        new_acl_json = self.RunGsUtil(['acl', 'get', suri(bucket1_uri, file_name)], return_stdout=True)
        self.assertEqual(public_read_acl, new_acl_json)
        resumable_size = ONE_KIB
        boto_config_for_test = ('GSUtil', 'resumable_threshold', str(resumable_size))
        with SetBotoConfigForTest([boto_config_for_test]):
            resumable_file_name = 'resumable_bar'
            resumable_contents = os.urandom(resumable_size)
            resumable_fpath = self.CreateTempFile(file_name=resumable_file_name, contents=resumable_contents)
            self.RunGsUtil(['cp', '-a', 'public-read', resumable_fpath, suri(bucket1_uri)])
            new_resumable_acl_json = self.RunGsUtil(['acl', 'get', suri(bucket1_uri, resumable_file_name)], return_stdout=True)
            self.assertEqual(public_read_acl, new_resumable_acl_json)

    def test_cp_key_to_local_stream(self):
        bucket_uri = self.CreateBucket()
        contents = b'foo'
        key_uri = self.CreateObject(bucket_uri=bucket_uri, contents=contents)
        stdout = self.RunGsUtil(['cp', suri(key_uri), '-'], return_stdout=True)
        self.assertIn(contents, stdout.encode('ascii'))

    def test_cp_local_file_to_local_stream(self):
        contents = b'content'
        fpath = self.CreateTempFile(contents=contents)
        stdout = self.RunGsUtil(['cp', fpath, '-'], return_stdout=True)
        self.assertIn(contents, stdout.encode(UTF8))

    @SequentialAndParallelTransfer
    def test_cp_zero_byte_file(self):
        dst_bucket_uri = self.CreateBucket()
        src_dir = self.CreateTempDir()
        fpath = os.path.join(src_dir, 'zero_byte')
        with open(fpath, 'w') as unused_out_file:
            pass
        self.RunGsUtil(['cp', fpath, suri(dst_bucket_uri)])

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check1():
            stdout = self.RunGsUtil(['ls', suri(dst_bucket_uri)], return_stdout=True)
            self.assertIn(os.path.basename(fpath), stdout)
        _Check1()
        download_path = os.path.join(src_dir, 'zero_byte_download')
        self.RunGsUtil(['cp', suri(dst_bucket_uri, 'zero_byte'), download_path])
        self.assertTrue(os.stat(download_path))

    def test_copy_bucket_to_bucket(self):
        """Tests recursively copying from bucket to bucket.

    This should produce identically named objects (and not, in particular,
    destination objects named by the version-specific URI from source objects).
    """
        src_bucket_uri = self.CreateVersionedBucket()
        dst_bucket_uri = self.CreateVersionedBucket()
        self.CreateObject(bucket_uri=src_bucket_uri, object_name='obj0', contents=b'abc')
        self.CreateObject(bucket_uri=src_bucket_uri, object_name='obj1', contents=b'def')

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _CopyAndCheck():
            self.RunGsUtil(['cp', '-R', suri(src_bucket_uri), suri(dst_bucket_uri)])
            stdout = self.RunGsUtil(['ls', '-R', dst_bucket_uri.uri], return_stdout=True)
            self.assertIn('%s%s/obj0\n' % (dst_bucket_uri, src_bucket_uri.bucket_name), stdout)
            self.assertIn('%s%s/obj1\n' % (dst_bucket_uri, src_bucket_uri.bucket_name), stdout)
        _CopyAndCheck()

    def test_copy_duplicate_nested_object_names_to_new_cloud_dir(self):
        """Tests copying from bucket to same bucket preserves file structure."""
        bucket_uri = self.CreateBucket()
        self.CreateObject(bucket_uri=bucket_uri, object_name='dir1/file.txt', contents=b'data')
        self.CreateObject(bucket_uri=bucket_uri, object_name='dir2/file.txt', contents=b'data')

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _CopyAndCheck():
            self.RunGsUtil(['cp', '-R', suri(bucket_uri) + '/*', suri(bucket_uri) + '/dst'])
            stdout = self.RunGsUtil(['ls', '-R', bucket_uri.uri], return_stdout=True)
            self.assertIn(suri(bucket_uri) + '/dst/dir1/file.txt', stdout)
            self.assertIn(suri(bucket_uri) + '/dst/dir2/file.txt', stdout)
        _CopyAndCheck()

    def test_copy_duplicate_nested_object_names_to_existing_cloud_dir(self):
        """Tests copying from bucket to same bucket preserves file structure."""
        bucket_uri = self.CreateBucket()
        self.CreateObject(bucket_uri=bucket_uri, object_name='dir1/file.txt', contents=b'data')
        self.CreateObject(bucket_uri=bucket_uri, object_name='dir2/file.txt', contents=b'data')
        self.CreateObject(bucket_uri=bucket_uri, object_name='dst/existing_file.txt', contents=b'data')

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _CopyAndCheck():
            self.RunGsUtil(['cp', '-R', suri(bucket_uri) + '/*', suri(bucket_uri) + '/dst'])
            stdout = self.RunGsUtil(['ls', '-R', bucket_uri.uri], return_stdout=True)
            self.assertIn(suri(bucket_uri) + '/dst/dir1/file.txt', stdout)
            self.assertIn(suri(bucket_uri) + '/dst/dir2/file.txt', stdout)
            self.assertIn(suri(bucket_uri) + '/dst/existing_file.txt', stdout)
        _CopyAndCheck()

    @SkipForGS('Only s3 V4 signatures error on location mismatches.')
    def test_copy_bucket_to_bucket_with_location_redirect(self):
        src_bucket_region = 'ap-east-1'
        dest_bucket_region = 'us-east-2'
        src_bucket_host = 's3.%s.amazonaws.com' % src_bucket_region
        dest_bucket_host = 's3.%s.amazonaws.com' % dest_bucket_region
        client_host = 's3.eu-west-1.amazonaws.com'
        with SetBotoConfigForTest([('s3', 'host', src_bucket_host)]):
            src_bucket_uri = self.CreateBucket(location=src_bucket_region)
            self.CreateObject(bucket_uri=src_bucket_uri, object_name='obj0', contents=b'abc')
            self.CreateObject(bucket_uri=src_bucket_uri, object_name='obj1', contents=b'def')
        with SetBotoConfigForTest([('s3', 'host', dest_bucket_host)]):
            dst_bucket_uri = self.CreateBucket(location=dest_bucket_region)

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _CopyAndCheck():
            self.RunGsUtil(['cp', '-R', suri(src_bucket_uri), suri(dst_bucket_uri)])
            stdout = self.RunGsUtil(['ls', '-R', dst_bucket_uri.uri], return_stdout=True)
            self.assertIn('%s%s/obj0\n' % (dst_bucket_uri, src_bucket_uri.bucket_name), stdout)
            self.assertIn('%s%s/obj1\n' % (dst_bucket_uri, src_bucket_uri.bucket_name), stdout)
        with SetBotoConfigForTest([('s3', 'host', client_host)]):
            _CopyAndCheck()

    def test_copy_bucket_to_dir(self):
        """Tests recursively copying from bucket to a directory.

    This should produce identically named objects (and not, in particular,
    destination objects named by the version- specific URI from source objects).
    """
        src_bucket_uri = self.CreateBucket()
        dst_dir = self.CreateTempDir()
        self.CreateObject(bucket_uri=src_bucket_uri, object_name='obj0', contents=b'abc')
        self.CreateObject(bucket_uri=src_bucket_uri, object_name='obj1', contents=b'def')

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _CopyAndCheck():
            """Copies the bucket recursively and validates the results."""
            self.RunGsUtil(['cp', '-R', suri(src_bucket_uri), dst_dir])
            dir_list = []
            for dirname, _, filenames in os.walk(dst_dir):
                for filename in filenames:
                    dir_list.append(os.path.join(dirname, filename))
            dir_list = sorted(dir_list)
            self.assertEqual(len(dir_list), 2)
            self.assertEqual(os.path.join(dst_dir, src_bucket_uri.bucket_name, 'obj0'), dir_list[0])
            self.assertEqual(os.path.join(dst_dir, src_bucket_uri.bucket_name, 'obj1'), dir_list[1])
        _CopyAndCheck()

    @unittest.skipUnless(HAS_S3_CREDS, 'Test requires both S3 and GS credentials')
    def test_copy_object_to_dir_s3_v4(self):
        """Tests copying object from s3 to local dir with v4 signature.

    Regions like us-east2 accept only V4 signature, hence we will create
    the bucket in us-east2 region to enforce testing with V4 signature.
    """
        src_bucket_uri = self.CreateBucket(provider='s3', location='us-east-2')
        dst_dir = self.CreateTempDir()
        self.CreateObject(bucket_uri=src_bucket_uri, object_name='obj0', contents=b'abc')
        self.CreateObject(bucket_uri=src_bucket_uri, object_name='obj1', contents=b'def')

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _CopyAndCheck():
            """Copies the bucket recursively and validates the results."""
            self.RunGsUtil(['cp', '-R', suri(src_bucket_uri), dst_dir])
            dir_list = []
            for dirname, _, filenames in os.walk(dst_dir):
                for filename in filenames:
                    dir_list.append(os.path.join(dirname, filename))
            dir_list = sorted(dir_list)
            self.assertEqual(len(dir_list), 2)
            self.assertEqual(os.path.join(dst_dir, src_bucket_uri.bucket_name, 'obj0'), dir_list[0])
            self.assertEqual(os.path.join(dst_dir, src_bucket_uri.bucket_name, 'obj1'), dir_list[1])
        _CopyAndCheck()

    @SkipForS3('The boto lib used for S3 does not handle objects starting with slashes if we use V4 signature')
    def test_recursive_download_with_leftover_slash_only_dir_placeholder(self):
        """Tests that we correctly handle leftover dir placeholders."""
        src_bucket_uri = self.CreateBucket()
        dst_dir = self.CreateTempDir()
        self.CreateObject(bucket_uri=src_bucket_uri, object_name='obj0', contents=b'abc')
        self.CreateObject(bucket_uri=src_bucket_uri, object_name='obj1', contents=b'def')
        key_uri = self.StorageUriCloneReplaceName(src_bucket_uri, '/')
        self.StorageUriSetContentsFromString(key_uri, '')
        self.AssertNObjectsInBucket(src_bucket_uri, 3)
        self.RunGsUtil(['cp', '-R', suri(src_bucket_uri), dst_dir])
        dir_list = []
        for dirname, _, filenames in os.walk(dst_dir):
            for filename in filenames:
                dir_list.append(os.path.join(dirname, filename))
        dir_list = sorted(dir_list)
        self.assertEqual(len(dir_list), 2)
        self.assertEqual(os.path.join(dst_dir, src_bucket_uri.bucket_name, 'obj0'), dir_list[0])
        self.assertEqual(os.path.join(dst_dir, src_bucket_uri.bucket_name, 'obj1'), dir_list[1])

    def test_recursive_download_with_leftover_dir_placeholder(self):
        """Tests that we correctly handle leftover dir placeholders."""
        src_bucket_uri = self.CreateBucket()
        dst_dir = self.CreateTempDir()
        self.CreateObject(bucket_uri=src_bucket_uri, object_name='obj0', contents=b'abc')
        self.CreateObject(bucket_uri=src_bucket_uri, object_name='obj1', contents=b'def')
        key_uri = self.StorageUriCloneReplaceName(src_bucket_uri, 'foo/')
        self.StorageUriSetContentsFromString(key_uri, '')
        self.AssertNObjectsInBucket(src_bucket_uri, 3)
        self.RunGsUtil(['cp', '-R', suri(src_bucket_uri), dst_dir])
        dir_list = []
        for dirname, _, filenames in os.walk(dst_dir):
            for filename in filenames:
                dir_list.append(os.path.join(dirname, filename))
        dir_list = sorted(dir_list)
        self.assertEqual(len(dir_list), 2)
        self.assertEqual(os.path.join(dst_dir, src_bucket_uri.bucket_name, 'obj0'), dir_list[0])
        self.assertEqual(os.path.join(dst_dir, src_bucket_uri.bucket_name, 'obj1'), dir_list[1])

    def test_copy_quiet(self):
        bucket_uri = self.CreateBucket()
        key_uri = self.CreateObject(bucket_uri=bucket_uri, contents=b'foo')
        stderr = self.RunGsUtil(['-q', 'cp', suri(key_uri), suri(self.StorageUriCloneReplaceName(bucket_uri, 'o2'))], return_stderr=True)
        self.assertEqual(stderr.count('Copying '), 0)

    def test_cp_md5_match(self):
        """Tests that the uploaded object has the expected MD5.

    Note that while this does perform a file to object upload, MD5's are
    not supported for composite objects so we don't use the decorator in this
    case.
    """
        bucket_uri = self.CreateBucket()
        fpath = self.CreateTempFile(contents=b'bar')
        with open(fpath, 'rb') as f_in:
            md5 = binascii.unhexlify(CalculateMd5FromContents(f_in))
            try:
                encoded_bytes = base64.encodebytes(md5)
            except AttributeError:
                encoded_bytes = base64.encodestring(md5)
            file_md5 = encoded_bytes.rstrip(b'\n')
        self.RunGsUtil(['cp', fpath, suri(bucket_uri)])

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check1():
            stdout = self.RunGsUtil(['ls', '-L', suri(bucket_uri)], return_stdout=True)
            self.assertRegex(stdout, 'Hash\\s+\\(md5\\):\\s+%s' % re.escape(file_md5.decode('ascii')))
        _Check1()

    @unittest.skipIf(IS_WINDOWS, 'Unicode handling on Windows requires mods to site-packages')
    @SequentialAndParallelTransfer
    def test_cp_manifest_upload_unicode(self):
        return self._ManifestUpload('foo-unicöde'.encode(UTF8), 'bar-unicöde'.encode(UTF8), 'manifest-unicöde'.encode(UTF8))

    @SequentialAndParallelTransfer
    def test_cp_manifest_upload(self):
        """Tests uploading with a mnifest file."""
        return self._ManifestUpload('foo', 'bar', 'manifest')

    def _ManifestUpload(self, file_name, object_name, manifest_name):
        """Tests uploading with a manifest file."""
        bucket_uri = self.CreateBucket()
        dsturi = suri(bucket_uri, object_name)
        fpath = self.CreateTempFile(file_name=file_name, contents=b'bar')
        logpath = self.CreateTempFile(file_name=manifest_name, contents=b'')
        open(logpath, 'w').close()
        self.RunGsUtil(['cp', '-L', logpath, fpath, dsturi])
        with open(logpath, 'r') as f:
            lines = f.readlines()
        if six.PY2:
            lines = [six.text_type(line, UTF8) for line in lines]
        self.assertEqual(len(lines), 2)
        expected_headers = ['Source', 'Destination', 'Start', 'End', 'Md5', 'UploadId', 'Source Size', 'Bytes Transferred', 'Result', 'Description']
        self.assertEqual(expected_headers, lines[0].strip().split(','))
        results = lines[1].strip().split(',')
        results = dict(zip(expected_headers, results))
        self.assertEqual(results['Source'], 'file://' + fpath)
        self.assertEqual(results['Destination'], dsturi)
        date_format = '%Y-%m-%dT%H:%M:%S.%fZ'
        start_date = datetime.datetime.strptime(results['Start'], date_format)
        end_date = datetime.datetime.strptime(results['End'], date_format)
        self.assertEqual(end_date > start_date, True)
        if self.RunGsUtil == testcase.GsUtilIntegrationTestCase.RunGsUtil:
            self.assertEqual(results['Md5'], 'rL0Y20zC+Fzt72VPzMSk2A==')
        self.assertEqual(int(results['Source Size']), 3)
        self.assertEqual(int(results['Bytes Transferred']), 3)
        self.assertEqual(results['Result'], 'OK')

    @SequentialAndParallelTransfer
    def test_cp_manifest_download(self):
        """Tests downloading with a manifest file."""
        key_uri = self.CreateObject(contents=b'foo')
        fpath = self.CreateTempFile(contents=b'')
        logpath = self.CreateTempFile(contents=b'')
        open(logpath, 'w').close()
        self.RunGsUtil(['cp', '-L', logpath, suri(key_uri), fpath], return_stdout=True)
        with open(logpath, 'r') as f:
            lines = f.readlines()
        if six.PY3:
            decode_lines = []
            for line in lines:
                if line.startswith("b'"):
                    some_strs = line.split(',')
                    line_parts = []
                    for some_str in some_strs:
                        if some_str.startswith("b'"):
                            line_parts.append(ast.literal_eval(some_str).decode(UTF8))
                        else:
                            line_parts.append(some_str)
                    decode_lines.append(','.join(line_parts))
                else:
                    decode_lines.append(line)
            lines = decode_lines
        self.assertEqual(len(lines), 2)
        expected_headers = ['Source', 'Destination', 'Start', 'End', 'Md5', 'UploadId', 'Source Size', 'Bytes Transferred', 'Result', 'Description']
        self.assertEqual(expected_headers, lines[0].strip().split(','))
        results = lines[1].strip().split(',')
        self.assertEqual(results[0][:5], '%s://' % self.default_provider)
        self.assertEqual(results[1][:7], 'file://')
        date_format = '%Y-%m-%dT%H:%M:%S.%fZ'
        start_date = datetime.datetime.strptime(results[2], date_format)
        end_date = datetime.datetime.strptime(results[3], date_format)
        self.assertEqual(end_date > start_date, True)
        self.assertEqual(int(results[6]), 3)
        self.assertGreaterEqual(int(results[7]), 3)
        self.assertEqual(results[8], 'OK')

    @SequentialAndParallelTransfer
    def test_copy_unicode_non_ascii_filename(self):
        key_uri = self.CreateObject()
        file_contents = b'x' * START_CALLBACK_PER_BYTES * 2
        fpath = self.CreateTempFile(file_name='Аудиоархив', contents=file_contents)
        with SetBotoConfigForTest([('GSUtil', 'resumable_threshold', '1')]):
            self.RunGsUtil(['cp', fpath, suri(key_uri)], return_stderr=True)
            stdout = self.RunGsUtil(['cat', suri(key_uri)], return_stdout=True, force_gsutil=True)
            self.assertEqual(stdout.encode('ascii'), file_contents)
        with SetBotoConfigForTest([('GSUtil', 'resumable_threshold', str(START_CALLBACK_PER_BYTES * 3))]):
            self.RunGsUtil(['cp', fpath, suri(key_uri)], return_stderr=True)
            stdout = self.RunGsUtil(['cat', suri(key_uri)], return_stdout=True, force_gsutil=True)
            self.assertEqual(stdout.encode('ascii'), file_contents)

    @SequentialAndParallelTransfer
    def test_gzip_upload_and_download(self):
        bucket_uri = self.CreateBucket()
        contents = b'x' * 10000
        tmpdir = self.CreateTempDir()
        self.CreateTempFile(file_name='test.html', tmpdir=tmpdir, contents=contents)
        self.CreateTempFile(file_name='test.js', tmpdir=tmpdir, contents=contents)
        self.CreateTempFile(file_name='test.txt', tmpdir=tmpdir, contents=contents)
        self.RunGsUtil(['cp', '-z', 'js, html', os.path.join(tmpdir, 'test.*'), suri(bucket_uri)])
        self.AssertNObjectsInBucket(bucket_uri, 3)
        uri1 = suri(bucket_uri, 'test.html')
        uri2 = suri(bucket_uri, 'test.js')
        uri3 = suri(bucket_uri, 'test.txt')
        stdout = self.RunGsUtil(['stat', uri1], return_stdout=True)
        self.assertRegex(stdout, 'Content-Encoding:\\s+gzip')
        stdout = self.RunGsUtil(['stat', uri2], return_stdout=True)
        self.assertRegex(stdout, 'Content-Encoding:\\s+gzip')
        stdout = self.RunGsUtil(['stat', uri3], return_stdout=True)
        self.assertNotRegex(stdout, 'Content-Encoding:\\s+gzip')
        fpath4 = self.CreateTempFile()
        for uri in (uri1, uri2, uri3):
            self.RunGsUtil(['cp', uri, suri(fpath4)])
            with open(fpath4, 'rb') as f:
                self.assertEqual(f.read(), contents)

    @SkipForS3('No compressed transport encoding support for S3.')
    @SkipForXML('No compressed transport encoding support for the XML API.')
    @SequentialAndParallelTransfer
    def test_gzip_transport_encoded_upload_and_download(self):
        """Test gzip encoded files upload correctly.

    This checks that files are not tagged with a gzip content encoding and
    that the contents of the files are uncompressed in GCS. This test uses the
    -j flag to target specific extensions.
    """

        def _create_test_data():
            """Setup the bucket and local data to test with.

      Returns:
        Triplet containing the following values:
          bucket_uri: String URI of cloud storage bucket to upload mock data
                      to.
          tmpdir: String, path of a temporary directory to write mock data to.
          local_uris: Tuple of three strings; each is the file path to a file
                      containing mock data.
      """
            bucket_uri = self.CreateBucket()
            contents = b'x' * 10000
            tmpdir = self.CreateTempDir()
            local_uris = []
            for filename in ('test.html', 'test.js', 'test.txt'):
                local_uris.append(self.CreateTempFile(file_name=filename, tmpdir=tmpdir, contents=contents))
            return (bucket_uri, tmpdir, local_uris)

        def _upload_test_data(tmpdir, bucket_uri):
            """Upload local test data.

      Args:
        tmpdir: String, path of a temporary directory to write mock data to.
        bucket_uri: String URI of cloud storage bucket to upload mock data to.

      Returns:
        stderr: String output from running the gsutil command to upload mock
                  data.
      """
            if self._use_gcloud_storage:
                extension_list_string = 'js,html'
            else:
                extension_list_string = 'js, html'
            stderr = self.RunGsUtil(['-D', 'cp', '-j', extension_list_string, os.path.join(tmpdir, 'test*'), suri(bucket_uri)], return_stderr=True)
            self.AssertNObjectsInBucket(bucket_uri, 3)
            return stderr

        def _assert_sent_compressed(local_uris, stderr):
            """Ensure the correct files were marked for compression.

      Args:
        local_uris: Tuple of three strings; each is the file path to a file
                    containing mock data.
        stderr: String output from running the gsutil command to upload mock
                data.
      """
            local_uri_html, local_uri_js, local_uri_txt = local_uris
            assert_base_string = 'Using compressed transport encoding for file://{}.'
            self.assertIn(assert_base_string.format(local_uri_html), stderr)
            self.assertIn(assert_base_string.format(local_uri_js), stderr)
            self.assertNotIn(assert_base_string.format(local_uri_txt), stderr)

        def _assert_stored_uncompressed(bucket_uri, contents=b'x' * 10000):
            """Ensure the files are not compressed when they are stored in the bucket.

      Args:
        bucket_uri: String with URI for bucket containing uploaded test data.
        contents: Byte string that are stored in each file in the bucket.
      """
            local_uri_html = suri(bucket_uri, 'test.html')
            local_uri_js = suri(bucket_uri, 'test.js')
            local_uri_txt = suri(bucket_uri, 'test.txt')
            fpath4 = self.CreateTempFile()
            for uri in (local_uri_html, local_uri_js, local_uri_txt):
                stdout = self.RunGsUtil(['stat', uri], return_stdout=True)
                self.assertNotRegex(stdout, 'Content-Encoding:\\s+gzip')
                self.RunGsUtil(['cp', uri, suri(fpath4)])
                with open(fpath4, 'rb') as f:
                    self.assertEqual(f.read(), contents)
        bucket_uri, tmpdir, local_uris = _create_test_data()
        stderr = _upload_test_data(tmpdir, bucket_uri)
        _assert_sent_compressed(local_uris, stderr)
        _assert_stored_uncompressed(bucket_uri)

    @SkipForS3('No compressed transport encoding support for S3.')
    @SkipForXML('No compressed transport encoding support for the XML API.')
    @SequentialAndParallelTransfer
    def test_gzip_transport_encoded_parallel_upload_non_resumable(self):
        """Test non resumable, gzip encoded files upload correctly in parallel.

    This test generates a small amount of data (e.g. 100 chars) to upload.
    Due to the small size, it will be below the resumable threshold,
    and test the behavior of non-resumable uploads.
    """
        bucket_uri = self.CreateBucket()
        contents = b'x' * 100
        tmpdir = self.CreateTempDir(test_files=10, contents=contents)
        with SetBotoConfigForTest([('GSUtil', 'resumable_threshold', str(ONE_KIB))]):
            stderr = self.RunGsUtil(['-D', '-m', 'cp', '-J', '-r', tmpdir, suri(bucket_uri)], return_stderr=True)
            self.AssertNObjectsInBucket(bucket_uri, 10)
            if not self._use_gcloud_storage:
                self.assertIn('send: Using gzip transport encoding for the request.', stderr)

    @SkipForS3('No compressed transport encoding support for S3.')
    @SkipForXML('No compressed transport encoding support for the XML API.')
    @SequentialAndParallelTransfer
    def test_gzip_transport_encoded_parallel_upload_resumable(self):
        """Test resumable, gzip encoded files upload correctly in parallel.

    This test generates a large amount of data (e.g. halt_size amount of chars)
    to upload. Due to the large size, it will be above the resumable threshold,
    and test the behavior of resumable uploads.
    """
        bucket_uri = self.CreateBucket()
        contents = get_random_ascii_chars(size=self.halt_size)
        tmpdir = self.CreateTempDir(test_files=10, contents=contents)
        with SetBotoConfigForTest([('GSUtil', 'resumable_threshold', str(ONE_KIB))]):
            stderr = self.RunGsUtil(['-D', '-m', 'cp', '-J', '-r', tmpdir, suri(bucket_uri)], return_stderr=True)
            self.AssertNObjectsInBucket(bucket_uri, 10)
            if not self._use_gcloud_storage:
                self.assertIn('send: Using gzip transport encoding for the request.', stderr)

    @SequentialAndParallelTransfer
    def test_gzip_all_upload_and_download(self):
        bucket_uri = self.CreateBucket()
        contents = b'x' * 10000
        tmpdir = self.CreateTempDir()
        self.CreateTempFile(file_name='test.html', tmpdir=tmpdir, contents=contents)
        self.CreateTempFile(file_name='test.js', tmpdir=tmpdir, contents=contents)
        self.CreateTempFile(file_name='test.txt', tmpdir=tmpdir, contents=contents)
        self.CreateTempFile(file_name='test', tmpdir=tmpdir, contents=contents)
        self.RunGsUtil(['cp', '-Z', os.path.join(tmpdir, 'test*'), suri(bucket_uri)])
        self.AssertNObjectsInBucket(bucket_uri, 4)
        uri1 = suri(bucket_uri, 'test.html')
        uri2 = suri(bucket_uri, 'test.js')
        uri3 = suri(bucket_uri, 'test.txt')
        uri4 = suri(bucket_uri, 'test')
        stdout = self.RunGsUtil(['stat', uri1], return_stdout=True)
        self.assertRegex(stdout, 'Content-Encoding:\\s+gzip')
        stdout = self.RunGsUtil(['stat', uri2], return_stdout=True)
        self.assertRegex(stdout, 'Content-Encoding:\\s+gzip')
        stdout = self.RunGsUtil(['stat', uri3], return_stdout=True)
        self.assertRegex(stdout, 'Content-Encoding:\\s+gzip')
        stdout = self.RunGsUtil(['stat', uri4], return_stdout=True)
        self.assertRegex(stdout, 'Content-Encoding:\\s+gzip')
        fpath4 = self.CreateTempFile()
        for uri in (uri1, uri2, uri3, uri4):
            self.RunGsUtil(['cp', uri, suri(fpath4)])
            with open(fpath4, 'rb') as f:
                self.assertEqual(f.read(), contents)

    @SkipForS3('No compressed transport encoding support for S3.')
    @SkipForXML('No compressed transport encoding support for the XML API.')
    @SequentialAndParallelTransfer
    def test_gzip_transport_encoded_all_upload_and_download(self):
        """Test gzip encoded files upload correctly.

    This checks that files are not tagged with a gzip content encoding and
    that the contents of the files are uncompressed in GCS. This test uses the
    -J flag to target all files.
    """
        bucket_uri = self.CreateBucket()
        contents = b'x' * 10000
        tmpdir = self.CreateTempDir()
        local_uri1 = self.CreateTempFile(file_name='test.txt', tmpdir=tmpdir, contents=contents)
        local_uri2 = self.CreateTempFile(file_name='test', tmpdir=tmpdir, contents=contents)
        stderr = self.RunGsUtil(['-D', 'cp', '-J', os.path.join(tmpdir, 'test*'), suri(bucket_uri)], return_stderr=True)
        self.AssertNObjectsInBucket(bucket_uri, 2)
        self.assertIn('Using compressed transport encoding for file://%s.' % local_uri1, stderr)
        self.assertIn('Using compressed transport encoding for file://%s.' % local_uri2, stderr)
        if not self._use_gcloud_storage:
            self.assertIn('send: Using gzip transport encoding for the request.', stderr)
        remote_uri1 = suri(bucket_uri, 'test.txt')
        remote_uri2 = suri(bucket_uri, 'test')
        fpath4 = self.CreateTempFile()
        for uri in (remote_uri1, remote_uri2):
            stdout = self.RunGsUtil(['stat', uri], return_stdout=True)
            self.assertNotRegex(stdout, 'Content-Encoding:\\s+gzip')
            self.RunGsUtil(['cp', uri, suri(fpath4)])
            with open(fpath4, 'rb') as f:
                self.assertEqual(f.read(), contents)

    def test_both_gzip_options_error(self):
        """Test that mixing compression flags error."""
        cases = (['cp', '-Z', '-z', 'html, js', 'a.js', 'b.js'], ['cp', '-z', 'html, js', '-Z', 'a.js', 'b.js'])
        if self._use_gcloud_storage:
            expected_status, expected_error_prefix, expected_error_substring = _GCLOUD_STORAGE_GZIP_FLAG_CONFLICT_OUTPUT
        else:
            expected_status = 1
            expected_error_prefix = 'CommandException'
            expected_error_substring = 'Specifying both the -z and -Z options together is invalid.'
        for case in cases:
            stderr = self.RunGsUtil(case, return_stderr=True, expected_status=expected_status)
            self.assertIn(expected_error_prefix, stderr)
            self.assertIn(expected_error_substring, stderr)

    def test_both_gzip_transport_encoding_options_error(self):
        """Test that mixing transport encoding flags error."""
        cases = (['cp', '-J', '-j', 'html, js', 'a.js', 'b.js'], ['cp', '-j', 'html, js', '-J', 'a.js', 'b.js'])
        if self._use_gcloud_storage:
            expected_status, expected_error_prefix, expected_error_substring = _GCLOUD_STORAGE_GZIP_FLAG_CONFLICT_OUTPUT
        else:
            expected_status = 1
            expected_error_prefix = 'CommandException'
            expected_error_substring = 'Specifying both the -j and -J options together is invalid.'
        for case in cases:
            stderr = self.RunGsUtil(case, return_stderr=True, expected_status=expected_status)
            self.assertIn(expected_error_prefix, stderr)
            self.assertIn(expected_error_substring, stderr)

    def test_combined_gzip_options_error(self):
        """Test that mixing transport encoding and compression flags error."""
        cases = (['cp', '-Z', '-j', 'html, js', 'a.js', 'b.js'], ['cp', '-J', '-z', 'html, js', 'a.js', 'b.js'], ['cp', '-j', 'html, js', '-Z', 'a.js', 'b.js'], ['cp', '-z', 'html, js', '-J', 'a.js', 'b.js'])
        if self._use_gcloud_storage:
            expected_status, expected_error_prefix, expected_error_substring = _GCLOUD_STORAGE_GZIP_FLAG_CONFLICT_OUTPUT
        else:
            expected_status = 1
            expected_error_prefix = 'CommandException'
            expected_error_substring = 'Specifying both the -j/-J and -z/-Z options together is invalid.'
        for case in cases:
            stderr = self.RunGsUtil(case, return_stderr=True, expected_status=expected_status)
            self.assertIn(expected_error_prefix, stderr)
            self.assertIn(expected_error_substring, stderr)

    def test_upload_with_subdir_and_unexpanded_wildcard(self):
        fpath1 = self.CreateTempFile(file_name=('tmp', 'x', 'y', 'z'))
        bucket_uri = self.CreateBucket()
        wildcard_uri = '%s*' % fpath1[:-5]
        stderr = self.RunGsUtil(['cp', '-R', wildcard_uri, suri(bucket_uri)], return_stderr=True)
        self.assertIn('Copying file:', stderr)
        self.AssertNObjectsInBucket(bucket_uri, 1)

    def test_upload_does_not_raise_with_content_md5_and_check_hashes_never(self):
        fpath1 = self.CreateTempFile(file_name='foo')
        bucket_uri = self.CreateBucket()
        with SetBotoConfigForTest([('GSUtil', 'check_hashes', 'never')]):
            stderr = self.RunGsUtil(['-h', 'Content-MD5: invalid-md5', 'cp', fpath1, suri(bucket_uri)], return_stderr=True)
            self.assertIn('Copying file:', stderr)
        self.AssertNObjectsInBucket(bucket_uri, 1)

    @SequentialAndParallelTransfer
    def test_cp_object_ending_with_slash(self):
        """Tests that cp works with object names ending with slash."""
        tmpdir = self.CreateTempDir()
        bucket_uri = self.CreateBucket()
        self.CreateObject(bucket_uri=bucket_uri, object_name='abc/', contents=b'dir')
        self.CreateObject(bucket_uri=bucket_uri, object_name='abc/def', contents=b'def')
        self.AssertNObjectsInBucket(bucket_uri, 2)
        self.RunGsUtil(['cp', '-R', suri(bucket_uri), tmpdir])
        with open(os.path.join(tmpdir, bucket_uri.bucket_name, 'abc', 'def')) as f:
            self.assertEqual('def', '\n'.join(f.readlines()))

    def test_cp_without_read_access(self):
        """Tests that cp fails without read access to the object."""
        bucket_uri = self.CreateBucket()
        object_uri = self.CreateObject(bucket_uri=bucket_uri, contents=b'foo')
        self.AssertNObjectsInBucket(bucket_uri, 1)
        if self.default_provider == 's3':
            expected_error_regex = 'AccessDenied'
        else:
            expected_error_regex = 'Anonymous \\S+ do(es)? not have'
        with self.SetAnonymousBotoCreds():
            stderr = self.RunGsUtil(['cp', suri(object_uri), 'foo'], return_stderr=True, expected_status=1)
        self.assertRegex(stderr, expected_error_regex)

    @unittest.skipIf(IS_WINDOWS, 'os.symlink() is not available on Windows.')
    def test_cp_minus_r_minus_e(self):
        """Tests that cp -e -r ignores symlinks when recursing."""
        bucket_uri = self.CreateBucket()
        tmpdir = self.CreateTempDir()
        self.CreateTempFile(tmpdir=tmpdir, contents=b'foo')
        subdir = os.path.join(tmpdir, 'subdir')
        os.mkdir(subdir)
        os.mkdir(os.path.join(tmpdir, 'missing'))
        os.symlink(os.path.join(tmpdir, 'missing'), os.path.join(subdir, 'missing'))
        os.rmdir(os.path.join(tmpdir, 'missing'))
        self.RunGsUtil(['cp', '-r', '-e', tmpdir, suri(bucket_uri)])

    @unittest.skipIf(IS_WINDOWS, 'os.symlink() is not available on Windows.')
    def test_cp_minus_e(self):
        fpath_dir = self.CreateTempDir()
        fpath1 = self.CreateTempFile(tmpdir=fpath_dir)
        fpath2 = os.path.join(fpath_dir, 'cp_minus_e')
        bucket_uri = self.CreateBucket()
        os.symlink(fpath1, fpath2)
        stderr = self.RunGsUtil(['-m', 'cp', '-e', '%s%s*' % (fpath_dir, os.path.sep), suri(bucket_uri, 'files')], return_stderr=True)
        self.assertIn('Copying file', stderr)
        if self._use_gcloud_storage:
            self.assertIn('Skipping symlink', stderr)
        else:
            self.assertIn('Skipping symbolic link', stderr)
        stderr = self.RunGsUtil(['cp', '-e', '-r', fpath1, fpath2, suri(bucket_uri, 'files')], return_stderr=True, expected_status=1)
        self.assertIn('Copying file', stderr)
        if self._use_gcloud_storage:
            self.assertIn('Skipping symlink', stderr)
            self.assertIn('URL matched no objects or files: %s' % fpath2, stderr)
        else:
            self.assertIn('Skipping symbolic link', stderr)
            self.assertIn('CommandException: No URLs matched: %s' % fpath2, stderr)

    def test_cp_multithreaded_wildcard(self):
        """Tests that cp -m works with a wildcard."""
        num_test_files = 5
        tmp_dir = self.CreateTempDir(test_files=num_test_files)
        bucket_uri = self.CreateBucket()
        wildcard_uri = '%s%s*' % (tmp_dir, os.sep)
        self.RunGsUtil(['-m', 'cp', wildcard_uri, suri(bucket_uri)])
        self.AssertNObjectsInBucket(bucket_uri, num_test_files)

    @SequentialAndParallelTransfer
    def test_cp_duplicate_source_args(self):
        """Tests that cp -m works when a source argument is provided twice."""
        object_contents = b'edge'
        object_uri = self.CreateObject(object_name='foo', contents=object_contents)
        tmp_dir = self.CreateTempDir()
        self.RunGsUtil(['-m', 'cp', suri(object_uri), suri(object_uri), tmp_dir])
        with open(os.path.join(tmp_dir, 'foo'), 'rb') as in_fp:
            contents = in_fp.read()
            self.assertEqual(contents, object_contents)

    @SkipForS3("gsutil doesn't support S3 customer-supplied encryption keys.")
    @SequentialAndParallelTransfer
    def test_cp_download_encrypted_object(self):
        """Tests downloading an encrypted object."""
        if self.test_api == ApiSelector.XML:
            return unittest.skip('gsutil does not support encryption with the XML API')
        object_contents = b'bar'
        object_uri = self.CreateObject(object_name='foo', contents=object_contents, encryption_key=TEST_ENCRYPTION_KEY1)
        fpath = self.CreateTempFile()
        boto_config_for_test = [('GSUtil', 'encryption_key', TEST_ENCRYPTION_KEY1)]
        with SetBotoConfigForTest(boto_config_for_test):
            self.RunGsUtil(['cp', suri(object_uri), suri(fpath)])
        with open(fpath, 'rb') as f:
            self.assertEqual(f.read(), object_contents)
        fpath2 = self.CreateTempFile()
        boto_config_for_test2 = [('GSUtil', 'encryption_key', TEST_ENCRYPTION_KEY3), ('GSUtil', 'decryption_key1', TEST_ENCRYPTION_KEY2), ('GSUtil', 'decryption_key2', TEST_ENCRYPTION_KEY1)]
        with SetBotoConfigForTest(boto_config_for_test2):
            self.RunGsUtil(['cp', suri(object_uri), suri(fpath2)])
        with open(fpath2, 'rb') as f:
            self.assertEqual(f.read(), object_contents)

    @SkipForS3("gsutil doesn't support S3 customer-supplied encryption keys.")
    @SequentialAndParallelTransfer
    def test_cp_download_encrypted_object_without_key(self):
        """Tests downloading an encrypted object without the necessary key."""
        if self.test_api == ApiSelector.XML:
            return unittest.skip('gsutil does not support encryption with the XML API')
        object_contents = b'bar'
        object_uri = self.CreateObject(object_name='foo', contents=object_contents, encryption_key=TEST_ENCRYPTION_KEY1)
        fpath = self.CreateTempFile()
        stderr = self.RunGsUtil(['cp', suri(object_uri), suri(fpath)], expected_status=1, return_stderr=True)
        self.assertIn('Missing decryption key with SHA256 hash %s' % TEST_ENCRYPTION_KEY1_SHA256_B64, stderr)

    @SkipForS3("gsutil doesn't support S3 customer-supplied encryption keys.")
    @SequentialAndParallelTransfer
    def test_cp_upload_encrypted_object(self):
        """Tests uploading an encrypted object."""
        if self.test_api == ApiSelector.XML:
            return unittest.skip('gsutil does not support encryption with the XML API')
        bucket_uri = self.CreateBucket()
        object_uri = suri(bucket_uri, 'foo')
        file_contents = b'bar'
        fpath = self.CreateTempFile(contents=file_contents, file_name='foo')
        boto_config_for_test = [('GSUtil', 'encryption_key', TEST_ENCRYPTION_KEY1)]
        with SetBotoConfigForTest(boto_config_for_test):
            self.RunGsUtil(['cp', suri(fpath), suri(bucket_uri)])
        self.AssertObjectUsesCSEK(object_uri, TEST_ENCRYPTION_KEY1)
        with SetBotoConfigForTest(boto_config_for_test):
            fpath2 = self.CreateTempFile()
            self.RunGsUtil(['cp', suri(bucket_uri, 'foo'), suri(fpath2)])
            with open(fpath2, 'rb') as f:
                self.assertEqual(f.read(), file_contents)

    @SkipForS3('No resumable upload or encryption support for S3.')
    def test_cp_resumable_upload_encrypted_object_break(self):
        """Tests that an encrypted upload resumes after a connection break."""
        if self.test_api == ApiSelector.XML:
            return unittest.skip('gsutil does not support encryption with the XML API')
        bucket_uri = self.CreateBucket()
        object_uri_str = suri(bucket_uri, 'foo')
        fpath = self.CreateTempFile(contents=b'a' * self.halt_size)
        boto_config_for_test = [('GSUtil', 'resumable_threshold', str(ONE_KIB)), ('GSUtil', 'encryption_key', TEST_ENCRYPTION_KEY1)]
        test_callback_file = self.CreateTempFile(contents=pickle.dumps(HaltingCopyCallbackHandler(True, 5)))
        with SetBotoConfigForTest(boto_config_for_test):
            stderr = self.RunGsUtil(['cp', '--testcallbackfile', test_callback_file, fpath, object_uri_str], expected_status=1, return_stderr=True)
            self.assertIn('Artifically halting upload', stderr)
            stderr = self.RunGsUtil(['cp', fpath, object_uri_str], return_stderr=True)
            self.assertIn('Resuming upload', stderr)
            stdout = self.RunGsUtil(['stat', object_uri_str], return_stdout=True)
            with open(fpath, 'rb') as fp:
                self.assertIn(CalculateB64EncodedMd5FromContents(fp), stdout)
        self.AssertObjectUsesCSEK(object_uri_str, TEST_ENCRYPTION_KEY1)

    @SkipForS3('No resumable upload or encryption support for S3.')
    def test_cp_resumable_upload_encrypted_object_different_key(self):
        """Tests that an encrypted upload resume uses original encryption key."""
        if self.test_api == ApiSelector.XML:
            return unittest.skip('gsutil does not support encryption with the XML API')
        bucket_uri = self.CreateBucket()
        object_uri_str = suri(bucket_uri, 'foo')
        file_contents = b'a' * self.halt_size
        fpath = self.CreateTempFile(contents=file_contents)
        boto_config_for_test = [('GSUtil', 'resumable_threshold', str(ONE_KIB)), ('GSUtil', 'encryption_key', TEST_ENCRYPTION_KEY1)]
        test_callback_file = self.CreateTempFile(contents=pickle.dumps(HaltingCopyCallbackHandler(True, 5)))
        with SetBotoConfigForTest(boto_config_for_test):
            stderr = self.RunGsUtil(['cp', '--testcallbackfile', test_callback_file, fpath, object_uri_str], expected_status=1, return_stderr=True)
            self.assertIn('Artifically halting upload', stderr)
        boto_config_for_test2 = [('GSUtil', 'resumable_threshold', str(ONE_KIB)), ('GSUtil', 'decryption_key1', TEST_ENCRYPTION_KEY2), ('GSUtil', 'encryption_key', TEST_ENCRYPTION_KEY1)]
        with SetBotoConfigForTest(boto_config_for_test2):
            stderr = self.RunGsUtil(['cp', fpath, object_uri_str], return_stderr=True)
            self.assertIn('Resuming upload', stderr)
        self.AssertObjectUsesCSEK(object_uri_str, TEST_ENCRYPTION_KEY1)

    @SkipForS3('No resumable upload or encryption support for S3.')
    def test_cp_resumable_upload_encrypted_object_missing_key(self):
        """Tests that an encrypted upload does not resume without original key."""
        if self.test_api == ApiSelector.XML:
            return unittest.skip('gsutil does not support encryption with the XML API')
        bucket_uri = self.CreateBucket()
        object_uri_str = suri(bucket_uri, 'foo')
        file_contents = b'a' * self.halt_size
        fpath = self.CreateTempFile(contents=file_contents)
        boto_config_for_test = [('GSUtil', 'resumable_threshold', str(ONE_KIB)), ('GSUtil', 'encryption_key', TEST_ENCRYPTION_KEY1)]
        test_callback_file = self.CreateTempFile(contents=pickle.dumps(HaltingCopyCallbackHandler(True, 5)))
        with SetBotoConfigForTest(boto_config_for_test):
            stderr = self.RunGsUtil(['cp', '--testcallbackfile', test_callback_file, fpath, object_uri_str], expected_status=1, return_stderr=True)
            self.assertIn('Artifically halting upload', stderr)
        boto_config_for_test2 = [('GSUtil', 'resumable_threshold', str(ONE_KIB)), ('GSUtil', 'encryption_key', TEST_ENCRYPTION_KEY2)]
        with SetBotoConfigForTest(boto_config_for_test2):
            stderr = self.RunGsUtil(['cp', fpath, object_uri_str], return_stderr=True)
            self.assertNotIn('Resuming upload', stderr)
            self.assertIn('does not match current encryption key', stderr)
            self.assertIn('Restarting upload from scratch', stderr)
            self.AssertObjectUsesCSEK(object_uri_str, TEST_ENCRYPTION_KEY2)

    def _ensure_object_unencrypted(self, object_uri_str):
        """Strongly consistent check that the object is unencrypted."""
        stdout = self.RunGsUtil(['stat', object_uri_str], return_stdout=True)
        self.assertNotIn('Encryption Key', stdout)

    @SkipForS3('No resumable upload support for S3.')
    def test_cp_resumable_upload_break(self):
        """Tests that an upload can be resumed after a connection break."""
        bucket_uri = self.CreateBucket()
        fpath = self.CreateTempFile(contents=b'a' * self.halt_size)
        boto_config_for_test = ('GSUtil', 'resumable_threshold', str(ONE_KIB))
        test_callback_file = self.CreateTempFile(contents=pickle.dumps(HaltingCopyCallbackHandler(True, 5)))
        with SetBotoConfigForTest([boto_config_for_test]):
            stderr = self.RunGsUtil(['cp', '--testcallbackfile', test_callback_file, fpath, suri(bucket_uri)], expected_status=1, return_stderr=True)
            self.assertIn('Artifically halting upload', stderr)
            stderr = self.RunGsUtil(['cp', fpath, suri(bucket_uri)], return_stderr=True)
            self.assertIn('Resuming upload', stderr)

    @SkipForS3('No compressed transport encoding support for S3.')
    @SkipForXML('No compressed transport encoding support for the XML API.')
    @SequentialAndParallelTransfer
    def test_cp_resumable_upload_gzip_encoded_break(self):
        """Tests that a gzip encoded upload can be resumed."""
        bucket_uri = self.CreateBucket()
        contents = get_random_ascii_chars(size=self.halt_size)
        local_uri = self.CreateTempFile(file_name='test.txt', contents=contents)
        boto_config_for_test = ('GSUtil', 'resumable_threshold', str(ONE_KIB))
        test_callback_file = self.CreateTempFile(contents=pickle.dumps(HaltingCopyCallbackHandler(True, 5)))
        with SetBotoConfigForTest([boto_config_for_test]):
            stderr = self.RunGsUtil(['-D', 'cp', '-J', '--testcallbackfile', test_callback_file, local_uri, suri(bucket_uri)], expected_status=1, return_stderr=True)
            self.assertIn('send: Using gzip transport encoding for the request.', stderr)
            self.assertIn('Artifically halting upload', stderr)
            stderr = self.RunGsUtil(['-D', 'cp', '-J', local_uri, suri(bucket_uri)], return_stderr=True)
            self.assertIn('Resuming upload', stderr)
            self.assertIn('send: Using gzip transport encoding for the request.', stderr)
        temp_uri = self.CreateTempFile()
        remote_uri = suri(bucket_uri, 'test.txt')
        stdout = self.RunGsUtil(['stat', remote_uri], return_stdout=True)
        self.assertNotRegex(stdout, 'Content-Encoding:\\s+gzip')
        self.RunGsUtil(['cp', remote_uri, suri(temp_uri)])
        with open(temp_uri, 'rb') as f:
            self.assertEqual(f.read(), contents)

    @SkipForS3('No resumable upload support for S3.')
    def test_cp_resumable_upload_retry(self):
        """Tests that a resumable upload completes with one retry."""
        bucket_uri = self.CreateBucket()
        fpath = self.CreateTempFile(contents=b'a' * self.halt_size)
        if self.test_api == ApiSelector.XML:
            test_callback_file = self.CreateTempFile(contents=pickle.dumps(_ResumableUploadRetryHandler(5, http_client.BadStatusLine, ('unused',))))
        else:
            test_callback_file = self.CreateTempFile(contents=pickle.dumps(_ResumableUploadRetryHandler(5, apitools_exceptions.BadStatusCodeError, ('unused', 'unused', 'unused'))))
        boto_config_for_test = ('GSUtil', 'resumable_threshold', str(ONE_KIB))
        with SetBotoConfigForTest([boto_config_for_test]):
            stderr = self.RunGsUtil(['-D', 'cp', '--testcallbackfile', test_callback_file, fpath, suri(bucket_uri)], return_stderr=1)
            if self.test_api == ApiSelector.XML:
                self.assertIn('Got retryable failure', stderr)
            else:
                self.assertIn('Retrying', stderr)

    @SkipForS3('No resumable upload support for S3.')
    def test_cp_resumable_streaming_upload_retry(self):
        """Tests that a streaming resumable upload completes with one retry."""
        if self.test_api == ApiSelector.XML:
            return unittest.skip('XML does not support resumable streaming uploads.')
        bucket_uri = self.CreateBucket()
        test_callback_file = self.CreateTempFile(contents=pickle.dumps(_ResumableUploadRetryHandler(5, apitools_exceptions.BadStatusCodeError, ('unused', 'unused', 'unused'))))
        boto_configs_for_test = [('GSUtil', 'json_resumable_chunk_size', str(256 * ONE_KIB)), ('Boto', 'num_retries', '2')]
        with SetBotoConfigForTest(boto_configs_for_test):
            stderr = self.RunGsUtil(['-D', 'cp', '--testcallbackfile', test_callback_file, '-', suri(bucket_uri, 'foo')], stdin='a' * 512 * ONE_KIB, return_stderr=1)
            self.assertIn('Retrying', stderr)

    @SkipForS3('preserve_acl flag not supported for S3.')
    def test_cp_preserve_no_owner(self):
        bucket_uri = self.CreateBucket()
        object_uri = self.CreateObject(bucket_uri=bucket_uri, contents=b'foo')
        self.RunGsUtil(['acl', 'ch', '-u', 'AllUsers:R', suri(object_uri)])
        self.RunGsUtil(['acl', 'ch', '-u', 'AllUsers:W', suri(bucket_uri)])
        with self.SetAnonymousBotoCreds():
            stderr = self.RunGsUtil(['cp', '-p', suri(object_uri), suri(bucket_uri, 'foo')], return_stderr=True, expected_status=1)
            self.assertIn('OWNER permission is required for preserving ACLs', stderr)

    @SkipForS3('No resumable upload support for S3.')
    def test_cp_progress_callbacks(self):
        bucket_uri = self.CreateBucket()
        final_size_string = BytesToFixedWidthString(1024 ** 2)
        final_progress_callback = final_size_string + '/' + final_size_string
        fpath = self.CreateTempFile(contents=b'a' * ONE_MIB, file_name='foo')
        boto_config_for_test = ('GSUtil', 'resumable_threshold', str(ONE_KIB))
        with SetBotoConfigForTest([boto_config_for_test]):
            stderr = self.RunGsUtil(['cp', fpath, suri(bucket_uri)], return_stderr=True)
            self.assertEqual(1, stderr.count(final_progress_callback))
        boto_config_for_test = ('GSUtil', 'resumable_threshold', str(2 * ONE_MIB))
        with SetBotoConfigForTest([boto_config_for_test]):
            stderr = self.RunGsUtil(['cp', fpath, suri(bucket_uri)], return_stderr=True)
            self.assertEqual(1, stderr.count(final_progress_callback))
        stderr = self.RunGsUtil(['cp', suri(bucket_uri, 'foo'), fpath], return_stderr=True)
        self.assertEqual(1, stderr.count(final_progress_callback))

    @SkipForS3('No resumable upload support for S3.')
    def test_cp_resumable_upload(self):
        """Tests that a basic resumable upload completes successfully."""
        bucket_uri = self.CreateBucket()
        fpath = self.CreateTempFile(contents=b'a' * self.halt_size)
        boto_config_for_test = ('GSUtil', 'resumable_threshold', str(ONE_KIB))
        with SetBotoConfigForTest([boto_config_for_test]):
            self.RunGsUtil(['cp', fpath, suri(bucket_uri)])

    @SkipForS3('No resumable upload support for S3.')
    def test_resumable_upload_break_leaves_tracker(self):
        """Tests that a tracker file is created with a resumable upload."""
        bucket_uri = self.CreateBucket()
        fpath = self.CreateTempFile(file_name='foo', contents=b'a' * self.halt_size)
        boto_config_for_test = ('GSUtil', 'resumable_threshold', str(ONE_KIB))
        with SetBotoConfigForTest([boto_config_for_test]):
            tracker_filename = GetTrackerFilePath(StorageUrlFromString(suri(bucket_uri, 'foo')), TrackerFileType.UPLOAD, self.test_api)
            test_callback_file = self.CreateTempFile(contents=pickle.dumps(HaltingCopyCallbackHandler(True, 5)))
            try:
                stderr = self.RunGsUtil(['cp', '--testcallbackfile', test_callback_file, fpath, suri(bucket_uri, 'foo')], expected_status=1, return_stderr=True)
                self.assertIn('Artifically halting upload', stderr)
                self.assertTrue(os.path.exists(tracker_filename), 'Tracker file %s not present.' % tracker_filename)
                if os.name == 'posix':
                    mode = oct(stat.S_IMODE(os.stat(tracker_filename).st_mode))
                    self.assertEqual(oct(384), mode)
            finally:
                DeleteTrackerFile(tracker_filename)

    @SkipForS3('No resumable upload support for S3.')
    def test_cp_resumable_upload_break_file_size_change(self):
        """Tests a resumable upload where the uploaded file changes size.

    This should fail when we read the tracker data.
    """
        bucket_uri = self.CreateBucket()
        tmp_dir = self.CreateTempDir()
        fpath = self.CreateTempFile(file_name='foo', tmpdir=tmp_dir, contents=b'a' * self.halt_size)
        test_callback_file = self.CreateTempFile(contents=pickle.dumps(HaltingCopyCallbackHandler(True, 5)))
        boto_config_for_test = ('GSUtil', 'resumable_threshold', str(ONE_KIB))
        with SetBotoConfigForTest([boto_config_for_test]):
            stderr = self.RunGsUtil(['cp', '--testcallbackfile', test_callback_file, fpath, suri(bucket_uri)], expected_status=1, return_stderr=True)
            self.assertIn('Artifically halting upload', stderr)
            fpath = self.CreateTempFile(file_name='foo', tmpdir=tmp_dir, contents=b'a' * self.halt_size * 2)
            stderr = self.RunGsUtil(['cp', fpath, suri(bucket_uri)], expected_status=1, return_stderr=True)
            self.assertIn('ResumableUploadAbortException', stderr)

    @SkipForS3('No resumable upload support for S3.')
    def test_cp_resumable_upload_break_file_content_change(self):
        """Tests a resumable upload where the uploaded file changes content."""
        if self.test_api == ApiSelector.XML:
            return unittest.skip("XML doesn't make separate HTTP calls at fixed-size boundaries for resumable uploads, so we can't guarantee that the server saves a specific part of the upload.")
        bucket_uri = self.CreateBucket()
        tmp_dir = self.CreateTempDir()
        fpath = self.CreateTempFile(file_name='foo', tmpdir=tmp_dir, contents=b'a' * ONE_KIB * ONE_KIB)
        test_callback_file = self.CreateTempFile(contents=pickle.dumps(HaltingCopyCallbackHandler(True, int(ONE_KIB) * 512)))
        resumable_threshold_for_test = ('GSUtil', 'resumable_threshold', str(ONE_KIB))
        resumable_chunk_size_for_test = ('GSUtil', 'json_resumable_chunk_size', str(ONE_KIB * 256))
        with SetBotoConfigForTest([resumable_threshold_for_test, resumable_chunk_size_for_test]):
            stderr = self.RunGsUtil(['cp', '--testcallbackfile', test_callback_file, fpath, suri(bucket_uri)], expected_status=1, return_stderr=True)
            self.assertIn('Artifically halting upload', stderr)
            fpath = self.CreateTempFile(file_name='foo', tmpdir=tmp_dir, contents=b'b' * ONE_KIB * ONE_KIB)
            stderr = self.RunGsUtil(['cp', fpath, suri(bucket_uri)], expected_status=1, return_stderr=True)
            self.assertIn("doesn't match cloud-supplied digest", stderr)

    @SkipForS3('No resumable upload support for S3.')
    def test_cp_resumable_upload_break_file_smaller_size(self):
        """Tests a resumable upload where the uploaded file changes content.

    This should fail hash validation.
    """
        bucket_uri = self.CreateBucket()
        tmp_dir = self.CreateTempDir()
        fpath = self.CreateTempFile(file_name='foo', tmpdir=tmp_dir, contents=b'a' * ONE_KIB * ONE_KIB)
        test_callback_file = self.CreateTempFile(contents=pickle.dumps(HaltingCopyCallbackHandler(True, int(ONE_KIB) * 512)))
        resumable_threshold_for_test = ('GSUtil', 'resumable_threshold', str(ONE_KIB))
        resumable_chunk_size_for_test = ('GSUtil', 'json_resumable_chunk_size', str(ONE_KIB * 256))
        with SetBotoConfigForTest([resumable_threshold_for_test, resumable_chunk_size_for_test]):
            stderr = self.RunGsUtil(['cp', '--testcallbackfile', test_callback_file, fpath, suri(bucket_uri)], expected_status=1, return_stderr=True)
            self.assertIn('Artifically halting upload', stderr)
            fpath = self.CreateTempFile(file_name='foo', tmpdir=tmp_dir, contents=b'a' * ONE_KIB)
            stderr = self.RunGsUtil(['cp', fpath, suri(bucket_uri)], expected_status=1, return_stderr=True)
            self.assertIn('ResumableUploadAbortException', stderr)

    @SkipForS3('No resumable upload support for S3.')
    def test_cp_composite_encrypted_upload_resume(self):
        """Tests that an encrypted composite upload resumes successfully."""
        if self.test_api == ApiSelector.XML:
            return unittest.skip('gsutil does not support encryption with the XML API')
        bucket_uri = self.CreateBucket()
        dst_url = StorageUrlFromString(suri(bucket_uri, 'foo'))
        file_contents = b'foobar'
        file_name = 'foobar'
        source_file = self.CreateTempFile(contents=file_contents, file_name=file_name)
        src_url = StorageUrlFromString(source_file)
        tracker_file_name = GetTrackerFilePath(dst_url, TrackerFileType.PARALLEL_UPLOAD, self.test_api, src_url)
        tracker_prefix = '123'
        encoded_name = (PARALLEL_UPLOAD_STATIC_SALT + source_file).encode(UTF8)
        content_md5 = GetMd5()
        content_md5.update(encoded_name)
        digest = content_md5.hexdigest()
        component_object_name = tracker_prefix + PARALLEL_UPLOAD_TEMP_NAMESPACE + digest + '_0'
        component_size = 3
        object_uri = self.CreateObject(bucket_uri=bucket_uri, object_name=component_object_name, contents=file_contents[:component_size], encryption_key=TEST_ENCRYPTION_KEY1)
        existing_component = ObjectFromTracker(component_object_name, str(object_uri.generation))
        existing_components = [existing_component]
        enc_key_sha256 = TEST_ENCRYPTION_KEY1_SHA256_B64
        WriteParallelUploadTrackerFile(tracker_file_name, tracker_prefix, existing_components, encryption_key_sha256=enc_key_sha256)
        try:
            with SetBotoConfigForTest([('GSUtil', 'parallel_composite_upload_threshold', '1'), ('GSUtil', 'parallel_composite_upload_component_size', str(component_size)), ('GSUtil', 'encryption_key', TEST_ENCRYPTION_KEY1)]):
                stderr = self.RunGsUtil(['cp', source_file, suri(bucket_uri, 'foo')], return_stderr=True)
                self.assertIn('Found 1 existing temporary components to reuse.', stderr)
                self.assertFalse(os.path.exists(tracker_file_name), 'Tracker file %s should have been deleted.' % tracker_file_name)
                read_contents = self.RunGsUtil(['cat', suri(bucket_uri, 'foo')], return_stdout=True)
                self.assertEqual(read_contents.encode('ascii'), file_contents)
        finally:
            DeleteTrackerFile(tracker_file_name)

    @SkipForS3('No resumable upload support for S3.')
    def test_cp_composite_encrypted_upload_restart(self):
        """Tests that encrypted composite upload restarts given a different key."""
        if self.test_api == ApiSelector.XML:
            return unittest.skip('gsutil does not support encryption with the XML API')
        bucket_uri = self.CreateBucket()
        dst_url = StorageUrlFromString(suri(bucket_uri, 'foo'))
        file_contents = b'foobar'
        source_file = self.CreateTempFile(contents=file_contents, file_name='foo')
        src_url = StorageUrlFromString(source_file)
        tracker_file_name = GetTrackerFilePath(dst_url, TrackerFileType.PARALLEL_UPLOAD, self.test_api, src_url)
        tracker_prefix = '123'
        existing_component_name = 'foo_1'
        object_uri = self.CreateObject(bucket_uri=bucket_uri, object_name='foo_1', contents=b'foo', encryption_key=TEST_ENCRYPTION_KEY1)
        existing_component = ObjectFromTracker(existing_component_name, str(object_uri.generation))
        existing_components = [existing_component]
        enc_key_sha256 = TEST_ENCRYPTION_KEY1_SHA256_B64
        WriteParallelUploadTrackerFile(tracker_file_name, tracker_prefix, existing_components, enc_key_sha256.decode('ascii'))
        try:
            with SetBotoConfigForTest([('GSUtil', 'parallel_composite_upload_threshold', '1'), ('GSUtil', 'parallel_composite_upload_component_size', '3'), ('GSUtil', 'encryption_key', TEST_ENCRYPTION_KEY2)]):
                stderr = self.RunGsUtil(['cp', source_file, suri(bucket_uri, 'foo')], return_stderr=True)
                self.assertIn('does not match current encryption key. Deleting old components and restarting upload', stderr)
                self.assertNotIn('existing temporary components to reuse.', stderr)
                self.assertFalse(os.path.exists(tracker_file_name), 'Tracker file %s should have been deleted.' % tracker_file_name)
                read_contents = self.RunGsUtil(['cat', suri(bucket_uri, 'foo')], return_stdout=True)
                self.assertEqual(read_contents.encode('ascii'), file_contents)
        finally:
            DeleteTrackerFile(tracker_file_name)

    @SkipForS3('Test uses gs-specific KMS encryption')
    def test_kms_key_correctly_applied_to_composite_upload(self):
        bucket_uri = self.CreateBucket()
        fpath = self.CreateTempFile(contents=b'abcd')
        obj_suri = suri(bucket_uri, 'composed')
        key_fqn = AuthorizeProjectToUseTestingKmsKey()
        with SetBotoConfigForTest([('GSUtil', 'encryption_key', key_fqn), ('GSUtil', 'parallel_composite_upload_threshold', '1'), ('GSUtil', 'parallel_composite_upload_component_size', '1')]):
            self.RunGsUtil(['cp', fpath, obj_suri])
        with SetBotoConfigForTest([('GSUtil', 'prefer_api', 'json')]):
            self.AssertObjectUsesCMEK(obj_suri, key_fqn)

    @SkipForS3('No composite upload support for S3.')
    def test_nearline_applied_to_parallel_composite_upload(self):
        bucket_uri = self.CreateBucket(storage_class='standard')
        fpath = self.CreateTempFile(contents=b'abcd')
        obj_suri = suri(bucket_uri, 'composed')
        with SetBotoConfigForTest([('GSUtil', 'parallel_composite_upload_threshold', '1'), ('GSUtil', 'parallel_composite_upload_component_size', '1')]):
            self.RunGsUtil(['cp', '-s', 'nearline', fpath, obj_suri])
        stdout = self.RunGsUtil(['ls', '-L', obj_suri], return_stdout=True)
        if self._use_gcloud_storage:
            self.assertRegexpMatchesWithFlags(stdout, 'Storage class:               NEARLINE', flags=re.IGNORECASE)
        else:
            self.assertRegexpMatchesWithFlags(stdout, 'Storage class:          NEARLINE', flags=re.IGNORECASE)

    @NotParallelizable
    @SkipForS3('No resumable upload support for S3.')
    @unittest.skipIf(IS_WINDOWS, 'chmod on dir unsupported on Windows.')
    @SequentialAndParallelTransfer
    def test_cp_unwritable_tracker_file(self):
        """Tests a resumable upload with an unwritable tracker file."""
        bucket_uri = self.CreateBucket()
        tracker_filename = GetTrackerFilePath(StorageUrlFromString(suri(bucket_uri, 'foo')), TrackerFileType.UPLOAD, self.test_api)
        tracker_dir = os.path.dirname(tracker_filename)
        fpath = self.CreateTempFile(file_name='foo', contents=b'a' * ONE_KIB)
        boto_config_for_test = ('GSUtil', 'resumable_threshold', str(ONE_KIB))
        save_mod = os.stat(tracker_dir).st_mode
        try:
            os.chmod(tracker_dir, 0)
            with SetBotoConfigForTest([boto_config_for_test]):
                stderr = self.RunGsUtil(['cp', fpath, suri(bucket_uri)], expected_status=1, return_stderr=True)
                self.assertIn("Couldn't write tracker file", stderr)
        finally:
            os.chmod(tracker_dir, save_mod)
            if os.path.exists(tracker_filename):
                os.unlink(tracker_filename)

    @NotParallelizable
    @unittest.skipIf(IS_WINDOWS, 'chmod on dir unsupported on Windows.')
    @SequentialAndParallelTransfer
    def test_cp_unwritable_tracker_file_download(self):
        """Tests downloads with an unwritable tracker file."""
        object_uri = self.CreateObject(contents=b'foo' * ONE_KIB)
        tracker_filename = GetTrackerFilePath(StorageUrlFromString(suri(object_uri)), TrackerFileType.DOWNLOAD, self.test_api)
        tracker_dir = os.path.dirname(tracker_filename)
        fpath = self.CreateTempFile()
        save_mod = os.stat(tracker_dir).st_mode
        try:
            os.chmod(tracker_dir, 0)
            boto_config_for_test = ('GSUtil', 'resumable_threshold', str(EIGHT_MIB))
            with SetBotoConfigForTest([boto_config_for_test]):
                self.RunGsUtil(['cp', suri(object_uri), fpath])
            boto_config_for_test = ('GSUtil', 'resumable_threshold', str(ONE_KIB))
            with SetBotoConfigForTest([boto_config_for_test]):
                stderr = self.RunGsUtil(['cp', suri(object_uri), fpath], expected_status=1, return_stderr=True)
                self.assertIn("Couldn't write tracker file", stderr)
        finally:
            os.chmod(tracker_dir, save_mod)
            if os.path.exists(tracker_filename):
                os.unlink(tracker_filename)

    def _test_cp_resumable_download_break_helper(self, boto_config, encryption_key=None):
        """Helper function for different modes of resumable download break.

    Args:
      boto_config: List of boto configuration tuples for use with
          SetBotoConfigForTest.
      encryption_key: Base64 encryption key for object encryption (if any).
    """
        bucket_uri = self.CreateBucket()
        file_contents = b'a' * self.halt_size
        object_uri = self.CreateObject(bucket_uri=bucket_uri, object_name='foo', contents=file_contents, encryption_key=encryption_key)
        fpath = self.CreateTempFile()
        test_callback_file = self.CreateTempFile(contents=pickle.dumps(HaltingCopyCallbackHandler(False, 5)))
        with SetBotoConfigForTest(boto_config):
            stderr = self.RunGsUtil(['cp', '--testcallbackfile', test_callback_file, suri(object_uri), fpath], expected_status=1, return_stderr=True)
            self.assertIn('Artifically halting download.', stderr)
            tracker_filename = GetTrackerFilePath(StorageUrlFromString(fpath), TrackerFileType.DOWNLOAD, self.test_api)
            self.assertTrue(os.path.isfile(tracker_filename))
            stderr = self.RunGsUtil(['cp', suri(object_uri), fpath], return_stderr=True)
            self.assertIn('Resuming download', stderr)
        with open(fpath, 'rb') as f:
            self.assertEqual(f.read(), file_contents, 'File contents differ')

    def test_cp_resumable_download_break(self):
        """Tests that a download can be resumed after a connection break."""
        self._test_cp_resumable_download_break_helper([('GSUtil', 'resumable_threshold', str(ONE_KIB))])

    @SkipForS3("gsutil doesn't support S3 customer-supplied encryption keys.")
    def test_cp_resumable_encrypted_download_break(self):
        """Tests that an encrypted download resumes after a connection break."""
        if self.test_api == ApiSelector.XML:
            return unittest.skip('gsutil does not support encryption with the XML API')
        self._test_cp_resumable_download_break_helper([('GSUtil', 'resumable_threshold', str(ONE_KIB)), ('GSUtil', 'encryption_key', TEST_ENCRYPTION_KEY1)], encryption_key=TEST_ENCRYPTION_KEY1)

    @SkipForS3("gsutil doesn't support S3 customer-supplied encryption keys.")
    def test_cp_resumable_encrypted_download_key_rotation(self):
        """Tests that a download restarts with a rotated encryption key."""
        if self.test_api == ApiSelector.XML:
            return unittest.skip('gsutil does not support encryption with the XML API')
        bucket_uri = self.CreateBucket()
        file_contents = b'a' * self.halt_size
        object_uri = self.CreateObject(bucket_uri=bucket_uri, object_name='foo', contents=file_contents, encryption_key=TEST_ENCRYPTION_KEY1)
        fpath = self.CreateTempFile()
        test_callback_file = self.CreateTempFile(contents=pickle.dumps(HaltingCopyCallbackHandler(False, 5)))
        boto_config_for_test = [('GSUtil', 'resumable_threshold', str(ONE_KIB)), ('GSUtil', 'encryption_key', TEST_ENCRYPTION_KEY1)]
        with SetBotoConfigForTest(boto_config_for_test):
            stderr = self.RunGsUtil(['cp', '--testcallbackfile', test_callback_file, suri(object_uri), fpath], expected_status=1, return_stderr=True)
            self.assertIn('Artifically halting download.', stderr)
            tracker_filename = GetTrackerFilePath(StorageUrlFromString(fpath), TrackerFileType.DOWNLOAD, self.test_api)
            self.assertTrue(os.path.isfile(tracker_filename))
        boto_config_for_test2 = [('GSUtil', 'resumable_threshold', str(ONE_KIB)), ('GSUtil', 'decryption_key1', TEST_ENCRYPTION_KEY1), ('GSUtil', 'encryption_key', TEST_ENCRYPTION_KEY2)]
        with SetBotoConfigForTest(boto_config_for_test2):
            self.RunGsUtil(['rewrite', '-k', suri(object_uri)])
        boto_config_for_test3 = [('GSUtil', 'resumable_threshold', str(ONE_KIB)), ('GSUtil', 'encryption_key', TEST_ENCRYPTION_KEY2)]
        with SetBotoConfigForTest(boto_config_for_test3):
            stderr = self.RunGsUtil(['cp', suri(object_uri), fpath], return_stderr=True)
            self.assertIn('Restarting download', stderr)
        with open(fpath, 'rb') as f:
            self.assertEqual(f.read(), file_contents, 'File contents differ')

    @SequentialAndParallelTransfer
    def test_cp_resumable_download_etag_differs(self):
        """Tests that download restarts the file when the source object changes.

    This causes the etag not to match.
    """
        bucket_uri = self.CreateBucket()
        object_uri = self.CreateObject(bucket_uri=bucket_uri, object_name='foo', contents=b'abc' * self.halt_size)
        fpath = self.CreateTempFile()
        test_callback_file = self.CreateTempFile(contents=pickle.dumps(HaltingCopyCallbackHandler(False, 5)))
        boto_config_for_test = ('GSUtil', 'resumable_threshold', str(ONE_KIB))
        with SetBotoConfigForTest([boto_config_for_test]):
            stderr = self.RunGsUtil(['cp', '--testcallbackfile', test_callback_file, suri(object_uri), fpath], expected_status=1, return_stderr=True)
            self.assertIn('Artifically halting download.', stderr)
            object_uri = self.CreateObject(bucket_uri=bucket_uri, object_name='foo', contents=b'b' * self.halt_size, gs_idempotent_generation=object_uri.generation)
            stderr = self.RunGsUtil(['cp', suri(object_uri), fpath], return_stderr=True)
            self.assertNotIn('Resuming download', stderr)

    @unittest.skipUnless(UsingCrcmodExtension(), 'Sliced download requires fast crcmod.')
    @SkipForS3('No sliced download support for S3.')
    def test_cp_resumable_download_generation_differs(self):
        """Tests that a resumable download restarts if the generation differs."""
        bucket_uri = self.CreateBucket()
        file_contents = b'abcd' * self.halt_size
        object_uri = self.CreateObject(bucket_uri=bucket_uri, object_name='foo', contents=file_contents)
        fpath = self.CreateTempFile()
        test_callback_file = self.CreateTempFile(contents=pickle.dumps(HaltingCopyCallbackHandler(False, 5)))
        boto_config_for_test = [('GSUtil', 'resumable_threshold', str(self.halt_size)), ('GSUtil', 'sliced_object_download_threshold', str(self.halt_size)), ('GSUtil', 'sliced_object_download_max_components', '3')]
        with SetBotoConfigForTest(boto_config_for_test):
            stderr = self.RunGsUtil(['cp', '--testcallbackfile', test_callback_file, suri(object_uri), suri(fpath)], return_stderr=True, expected_status=1)
            self.assertIn('Artifically halting download.', stderr)
            identical_file = self.CreateTempFile(contents=file_contents)
            self.RunGsUtil(['cp', suri(identical_file), suri(object_uri)])
            stderr = self.RunGsUtil(['cp', suri(object_uri), suri(fpath)], return_stderr=True)
            self.assertIn('Restarting download from scratch', stderr)
            with open(fpath, 'rb') as f:
                self.assertEqual(f.read(), file_contents, 'File contents differ')

    def test_cp_resumable_download_file_larger(self):
        """Tests download deletes the tracker file when existing file is larger."""
        bucket_uri = self.CreateBucket()
        fpath = self.CreateTempFile()
        object_uri = self.CreateObject(bucket_uri=bucket_uri, object_name='foo', contents=b'a' * self.halt_size)
        test_callback_file = self.CreateTempFile(contents=pickle.dumps(HaltingCopyCallbackHandler(False, 5)))
        boto_config_for_test = ('GSUtil', 'resumable_threshold', str(ONE_KIB))
        with SetBotoConfigForTest([boto_config_for_test]):
            stderr = self.RunGsUtil(['cp', '--testcallbackfile', test_callback_file, suri(object_uri), fpath], expected_status=1, return_stderr=True)
            self.assertIn('Artifically halting download.', stderr)
            with open(fpath + '_.gstmp', 'w') as larger_file:
                for _ in range(self.halt_size * 2):
                    larger_file.write('a')
            stderr = self.RunGsUtil(['cp', suri(object_uri), fpath], expected_status=1, return_stderr=True)
            self.assertNotIn('Resuming download', stderr)
            self.assertIn('Deleting tracker file', stderr)

    def test_cp_resumable_download_content_differs(self):
        """Tests that we do not re-download when tracker file matches existing file.

    We only compare size, not contents, so re-download should not occur even
    though the contents are technically different. However, hash validation on
    the file should still occur and we will delete the file then because
    the hashes differ.
    """
        bucket_uri = self.CreateBucket()
        tmp_dir = self.CreateTempDir()
        fpath = self.CreateTempFile(tmpdir=tmp_dir)
        temp_download_file = fpath + '_.gstmp'
        with open(temp_download_file, 'w') as fp:
            fp.write('abcd' * ONE_KIB)
        object_uri = self.CreateObject(bucket_uri=bucket_uri, object_name='foo', contents=b'efgh' * ONE_KIB)
        stdout = self.RunGsUtil(['ls', '-L', suri(object_uri)], return_stdout=True)
        etag_match = re.search('\\s*ETag:\\s*(.*)', stdout)
        self.assertIsNotNone(etag_match, 'Could not get object ETag')
        self.assertEqual(len(etag_match.groups()), 1, 'Did not match expected single ETag')
        etag = etag_match.group(1)
        tracker_filename = GetTrackerFilePath(StorageUrlFromString(fpath), TrackerFileType.DOWNLOAD, self.test_api)
        try:
            with open(tracker_filename, 'w') as tracker_fp:
                tracker_fp.write(etag)
            boto_config_for_test = ('GSUtil', 'resumable_threshold', str(ONE_KIB))
            with SetBotoConfigForTest([boto_config_for_test]):
                stderr = self.RunGsUtil(['cp', suri(object_uri), fpath], return_stderr=True, expected_status=1)
                self.assertIn('Download already complete', stderr)
                self.assertIn("doesn't match cloud-supplied digest", stderr)
                self.assertFalse(os.path.isfile(temp_download_file))
                self.assertFalse(os.path.isfile(tracker_filename))
                self.assertFalse(os.path.isfile(fpath))
        finally:
            if os.path.exists(tracker_filename):
                os.unlink(tracker_filename)

    def test_cp_resumable_download_content_matches(self):
        """Tests download no-ops when tracker file matches existing file."""
        bucket_uri = self.CreateBucket()
        tmp_dir = self.CreateTempDir()
        fpath = self.CreateTempFile(tmpdir=tmp_dir)
        matching_contents = b'abcd' * ONE_KIB
        temp_download_file = fpath + '_.gstmp'
        with open(temp_download_file, 'wb') as fp:
            fp.write(matching_contents)
        object_uri = self.CreateObject(bucket_uri=bucket_uri, object_name='foo', contents=matching_contents)
        stdout = self.RunGsUtil(['ls', '-L', suri(object_uri)], return_stdout=True)
        etag_match = re.search('\\s*ETag:\\s*(.*)', stdout)
        self.assertIsNotNone(etag_match, 'Could not get object ETag')
        self.assertEqual(len(etag_match.groups()), 1, 'Did not match expected single ETag')
        etag = etag_match.group(1)
        tracker_filename = GetTrackerFilePath(StorageUrlFromString(fpath), TrackerFileType.DOWNLOAD, self.test_api)
        with open(tracker_filename, 'w') as tracker_fp:
            tracker_fp.write(etag)
        try:
            boto_config_for_test = ('GSUtil', 'resumable_threshold', str(ONE_KIB))
            with SetBotoConfigForTest([boto_config_for_test]):
                stderr = self.RunGsUtil(['cp', suri(object_uri), fpath], return_stderr=True)
                self.assertIn('Download already complete', stderr)
                self.assertFalse(os.path.isfile(tracker_filename))
        finally:
            if os.path.exists(tracker_filename):
                os.unlink(tracker_filename)

    def test_cp_resumable_download_tracker_file_not_matches(self):
        """Tests that download overwrites when tracker file etag does not match."""
        bucket_uri = self.CreateBucket()
        tmp_dir = self.CreateTempDir()
        fpath = self.CreateTempFile(tmpdir=tmp_dir, contents=b'abcd' * ONE_KIB)
        object_uri = self.CreateObject(bucket_uri=bucket_uri, object_name='foo', contents=b'efgh' * ONE_KIB)
        stdout = self.RunGsUtil(['ls', '-L', suri(object_uri)], return_stdout=True)
        etag_match = re.search('\\s*ETag:\\s*(.*)', stdout)
        self.assertIsNotNone(etag_match, 'Could not get object ETag')
        self.assertEqual(len(etag_match.groups()), 1, 'Did not match regex for exactly one object ETag')
        etag = etag_match.group(1)
        etag += 'nonmatching'
        tracker_filename = GetTrackerFilePath(StorageUrlFromString(fpath), TrackerFileType.DOWNLOAD, self.test_api)
        with open(tracker_filename, 'w') as tracker_fp:
            tracker_fp.write(etag)
        try:
            boto_config_for_test = ('GSUtil', 'resumable_threshold', str(ONE_KIB))
            with SetBotoConfigForTest([boto_config_for_test]):
                stderr = self.RunGsUtil(['cp', suri(object_uri), fpath], return_stderr=True)
                self.assertNotIn('Resuming download', stderr)
                with open(fpath, 'r') as in_fp:
                    contents = in_fp.read()
                    self.assertEqual(contents, 'efgh' * ONE_KIB, 'File not overwritten when it should have been due to a non-matching tracker file.')
                self.assertFalse(os.path.isfile(tracker_filename))
        finally:
            if os.path.exists(tracker_filename):
                os.unlink(tracker_filename)

    def test_cp_double_gzip(self):
        """Tests that upload and download of a doubly-gzipped file succeeds."""
        bucket_uri = self.CreateBucket()
        fpath = self.CreateTempFile(file_name='looks-zipped.gz', contents=b'foo')
        self.RunGsUtil(['-h', 'content-type:application/gzip', 'cp', '-Z', suri(fpath), suri(bucket_uri, 'foo')])
        self.RunGsUtil(['cp', suri(bucket_uri, 'foo'), fpath])

    @SkipForS3('No compressed transport encoding support for S3.')
    @SkipForXML('No compressed transport encoding support for the XML API.')
    @SequentialAndParallelTransfer
    def test_cp_double_gzip_transport_encoded(self):
        """Tests that upload and download of a doubly-gzipped file succeeds."""
        bucket_uri = self.CreateBucket()
        fpath = self.CreateTempFile(file_name='looks-zipped.gz', contents=b'foo')
        stderr = self.RunGsUtil(['-DD', '-h', 'content-type:application/gzip', 'cp', '-J', suri(fpath), suri(bucket_uri, 'foo')], return_stderr=True)
        if self._use_gcloud_storage:
            self.assertIn("b'Content-Encoding': b'gzip'", stderr)
            self.assertIn('"contentType": "application/gzip"', stderr)
        else:
            self.assertIn("'Content-Encoding': 'gzip'", stderr)
            self.assertIn("contentType: 'application/gzip'", stderr)
        self.RunGsUtil(['cp', suri(bucket_uri, 'foo'), fpath])

    @unittest.skipIf(IS_WINDOWS, 'TODO(b/293885158) Timeout on Windows.')
    @SequentialAndParallelTransfer
    def test_cp_resumable_download_gzip(self):
        """Tests that download can be resumed successfully with a gzipped file."""
        object_uri = self.CreateObject()
        random.seed(0)
        contents = str([random.choice(string.ascii_letters) for _ in xrange(self.halt_size)]).encode('ascii')
        random.seed()
        fpath1 = self.CreateTempFile(file_name='unzipped.txt', contents=contents)
        self.RunGsUtil(['cp', '-z', 'txt', suri(fpath1), suri(object_uri)])

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _GetObjectSize():
            stdout = self.RunGsUtil(['du', suri(object_uri)], return_stdout=True)
            size_match = re.search('(\\d+)\\s+.*', stdout)
            self.assertIsNotNone(size_match, 'Could not get object size')
            self.assertEqual(len(size_match.groups()), 1, 'Did not match regex for exactly one object size.')
            return long(size_match.group(1))
        object_size = _GetObjectSize()
        self.assertGreaterEqual(object_size, self.halt_size, 'Compresed object size was not large enough to allow for a halted download, so the test results would be invalid. Please increase the compressed object size in the test.')
        fpath2 = self.CreateTempFile()
        test_callback_file = self.CreateTempFile(contents=pickle.dumps(HaltingCopyCallbackHandler(False, 5)))
        boto_config_for_test = ('GSUtil', 'resumable_threshold', str(ONE_KIB))
        with SetBotoConfigForTest([boto_config_for_test]):
            stderr = self.RunGsUtil(['cp', '--testcallbackfile', test_callback_file, suri(object_uri), suri(fpath2)], return_stderr=True, expected_status=1)
            self.assertIn('Artifically halting download.', stderr)
            self.assertIn('Downloading to temp gzip filename', stderr)
            sliced_download_threshold = HumanReadableToBytes(boto.config.get('GSUtil', 'sliced_object_download_threshold', DEFAULT_SLICED_OBJECT_DOWNLOAD_THRESHOLD))
            sliced_download = len(contents) > sliced_download_threshold and sliced_download_threshold > 0 and UsingCrcmodExtension()
            if sliced_download:
                trackerfile_type = TrackerFileType.SLICED_DOWNLOAD
            else:
                trackerfile_type = TrackerFileType.DOWNLOAD
            tracker_filename = GetTrackerFilePath(StorageUrlFromString(fpath2), trackerfile_type, self.test_api)
            self.assertTrue(os.path.isfile(tracker_filename))
            self.assertTrue(os.path.isfile('%s_.gztmp' % fpath2))
            stderr = self.RunGsUtil(['cp', suri(object_uri), suri(fpath2)], return_stderr=True)
            self.assertIn('Resuming download', stderr)
            with open(fpath2, 'rb') as f:
                self.assertEqual(f.read(), contents, 'File contents did not match.')
            self.assertFalse(os.path.isfile(tracker_filename))
            self.assertFalse(os.path.isfile('%s_.gztmp' % fpath2))

    def _GetFaviconFile(self):
        if not hasattr(self, 'test_data_favicon_file'):
            contents = pkgutil.get_data('gslib', 'tests/test_data/favicon.ico.gz')
            self.test_data_favicon_file = self.CreateTempFile(contents=contents)
        return self.test_data_favicon_file

    def test_cp_download_transfer_encoded(self):
        """Tests chunked transfer encoded download handling.

    Tests that download works correctly with a gzipped chunked transfer-encoded
    object (which therefore lacks Content-Length) of a size that gets fetched
    in a single chunk (exercising downloading of objects lacking a length
    response header).
    """
        bucket_uri = self.CreateBucket()
        object_uri = self.CreateObject(bucket_uri=bucket_uri, object_name='foo')
        input_filename = self._GetFaviconFile()
        self.RunGsUtil(['-h', 'Content-Encoding:gzip', '-h', 'Content-Type:image/x-icon', 'cp', suri(input_filename), suri(object_uri)])
        with gzip.open(input_filename) as fp:
            hash_dict = {'md5': GetMd5()}
            hashing_helper.CalculateHashesFromContents(fp, hash_dict)
            in_file_md5 = hash_dict['md5'].digest()
        fpath2 = self.CreateTempFile()
        self.RunGsUtil(['cp', suri(object_uri), suri(fpath2)])
        with open(fpath2, 'rb') as fp:
            hash_dict = {'md5': GetMd5()}
            hashing_helper.CalculateHashesFromContents(fp, hash_dict)
            out_file_md5 = hash_dict['md5'].digest()
        self.assertEqual(in_file_md5, out_file_md5)

    @SequentialAndParallelTransfer
    def test_cp_resumable_download_check_hashes_never(self):
        """Tests that resumble downloads work with check_hashes = never."""
        bucket_uri = self.CreateBucket()
        contents = b'abcd' * self.halt_size
        object_uri = self.CreateObject(bucket_uri=bucket_uri, object_name='foo', contents=contents)
        fpath = self.CreateTempFile()
        test_callback_file = self.CreateTempFile(contents=pickle.dumps(HaltingCopyCallbackHandler(False, 5)))
        boto_config_for_test = [('GSUtil', 'resumable_threshold', str(ONE_KIB)), ('GSUtil', 'check_hashes', 'never')]
        with SetBotoConfigForTest(boto_config_for_test):
            stderr = self.RunGsUtil(['cp', '--testcallbackfile', test_callback_file, suri(object_uri), fpath], expected_status=1, return_stderr=True)
            self.assertIn('Artifically halting download.', stderr)
            stderr = self.RunGsUtil(['cp', suri(object_uri), fpath], return_stderr=True)
            self.assertIn('Resuming download', stderr)
            self.assertIn('Found no hashes to validate object downloaded', stderr)
            with open(fpath, 'rb') as f:
                self.assertEqual(f.read(), contents, 'File contents did not match.')

    @SkipForS3('No resumable upload support for S3.')
    def test_cp_resumable_upload_bucket_deleted(self):
        """Tests that a not found exception is raised if bucket no longer exists."""
        bucket_uri = self.CreateBucket()
        fpath = self.CreateTempFile(contents=b'a' * 2 * ONE_KIB)
        boto_config_for_test = ('GSUtil', 'resumable_threshold', str(ONE_KIB))
        test_callback_file = self.CreateTempFile(contents=pickle.dumps(_DeleteBucketThenStartOverCopyCallbackHandler(5, bucket_uri)))
        with SetBotoConfigForTest([boto_config_for_test]):
            stderr = self.RunGsUtil(['cp', '--testcallbackfile', test_callback_file, fpath, suri(bucket_uri)], return_stderr=True, expected_status=1)
        self.assertIn('Deleting bucket', stderr)
        self.assertIn('bucket does not exist', stderr)

    @SkipForS3('No sliced download support for S3.')
    def test_cp_sliced_download(self):
        """Tests that sliced object download works in the general case."""
        bucket_uri = self.CreateBucket()
        object_uri = self.CreateObject(bucket_uri=bucket_uri, object_name='foo', contents=b'abc' * ONE_KIB)
        fpath = self.CreateTempFile()
        boto_config_for_test = [('GSUtil', 'resumable_threshold', str(ONE_KIB)), ('GSUtil', 'test_assume_fast_crcmod', 'True'), ('GSUtil', 'sliced_object_download_threshold', str(ONE_KIB)), ('GSUtil', 'sliced_object_download_max_components', '3')]
        with SetBotoConfigForTest(boto_config_for_test):
            self.RunGsUtil(['cp', suri(object_uri), fpath])
            tracker_filenames = GetSlicedDownloadTrackerFilePaths(StorageUrlFromString(fpath), self.test_api)
            for tracker_filename in tracker_filenames:
                self.assertFalse(os.path.isfile(tracker_filename))
            with open(fpath, 'rb') as f:
                self.assertEqual(f.read(), b'abc' * ONE_KIB, 'File contents differ')

    @unittest.skipUnless(UsingCrcmodExtension(), 'Sliced download requires fast crcmod.')
    @SkipForS3('No sliced download support for S3.')
    def test_cp_unresumable_sliced_download(self):
        """Tests sliced download works when resumability is disabled."""
        bucket_uri = self.CreateBucket()
        object_uri = self.CreateObject(bucket_uri=bucket_uri, object_name='foo', contents=b'abcd' * self.halt_size)
        fpath = self.CreateTempFile()
        test_callback_file = self.CreateTempFile(contents=pickle.dumps(HaltingCopyCallbackHandler(False, 5)))
        boto_config_for_test = [('GSUtil', 'resumable_threshold', str(self.halt_size * 5)), ('GSUtil', 'sliced_object_download_threshold', str(self.halt_size)), ('GSUtil', 'sliced_object_download_max_components', '4')]
        with SetBotoConfigForTest(boto_config_for_test):
            stderr = self.RunGsUtil(['cp', '--testcallbackfile', test_callback_file, suri(object_uri), suri(fpath)], return_stderr=True, expected_status=1)
            self.assertIn('not downloaded successfully', stderr)
            self.assertTrue(os.path.isfile(fpath + '_.gstmp'))
            tracker_filenames = GetSlicedDownloadTrackerFilePaths(StorageUrlFromString(fpath), self.test_api)
            for tracker_filename in tracker_filenames:
                self.assertFalse(os.path.isfile(tracker_filename))
        with SetBotoConfigForTest(boto_config_for_test):
            stderr = self.RunGsUtil(['cp', suri(object_uri), suri(fpath)], return_stderr=True)
            self.assertNotIn('Resuming download', stderr)
            self.assertFalse(os.path.isfile(fpath + '_.gstmp'))
            with open(fpath, 'rb') as f:
                self.assertEqual(f.read(), b'abcd' * self.halt_size, 'File contents differ')

    @unittest.skipUnless(UsingCrcmodExtension(), 'Sliced download requires fast crcmod.')
    @SkipForS3('No sliced download support for S3.')
    def test_cp_sliced_download_resume(self):
        """Tests that sliced object download is resumable."""
        bucket_uri = self.CreateBucket()
        object_uri = self.CreateObject(bucket_uri=bucket_uri, object_name='foo', contents=b'abc' * self.halt_size)
        fpath = self.CreateTempFile()
        test_callback_file = self.CreateTempFile(contents=pickle.dumps(HaltingCopyCallbackHandler(False, 5)))
        boto_config_for_test = [('GSUtil', 'resumable_threshold', str(self.halt_size)), ('GSUtil', 'sliced_object_download_threshold', str(self.halt_size)), ('GSUtil', 'sliced_object_download_max_components', '3')]
        with SetBotoConfigForTest(boto_config_for_test):
            stderr = self.RunGsUtil(['cp', '--testcallbackfile', test_callback_file, suri(object_uri), suri(fpath)], return_stderr=True, expected_status=1)
            self.assertIn('not downloaded successfully', stderr)
            tracker_filenames = GetSlicedDownloadTrackerFilePaths(StorageUrlFromString(fpath), self.test_api)
            for tracker_filename in tracker_filenames:
                self.assertTrue(os.path.isfile(tracker_filename))
            stderr = self.RunGsUtil(['cp', suri(object_uri), fpath], return_stderr=True)
            self.assertIn('Resuming download', stderr)
            tracker_filenames = GetSlicedDownloadTrackerFilePaths(StorageUrlFromString(fpath), self.test_api)
            for tracker_filename in tracker_filenames:
                self.assertFalse(os.path.isfile(tracker_filename))
            with open(fpath, 'rb') as f:
                self.assertEqual(f.read(), b'abc' * self.halt_size, 'File contents differ')

    @unittest.skipUnless(UsingCrcmodExtension(), 'Sliced download requires fast crcmod.')
    @SkipForS3('No sliced download support for S3.')
    def test_cp_sliced_download_partial_resume(self):
        """Test sliced download resumability when some components are finished."""
        bucket_uri = self.CreateBucket()
        object_uri = self.CreateObject(bucket_uri=bucket_uri, object_name='foo', contents=b'abc' * self.halt_size)
        fpath = self.CreateTempFile()
        test_callback_file = self.CreateTempFile(contents=pickle.dumps(HaltOneComponentCopyCallbackHandler(5)))
        boto_config_for_test = [('GSUtil', 'resumable_threshold', str(self.halt_size)), ('GSUtil', 'sliced_object_download_threshold', str(self.halt_size)), ('GSUtil', 'sliced_object_download_max_components', '3')]
        with SetBotoConfigForTest(boto_config_for_test):
            stderr = self.RunGsUtil(['cp', '--testcallbackfile', test_callback_file, suri(object_uri), suri(fpath)], return_stderr=True, expected_status=1)
            self.assertIn('not downloaded successfully', stderr)
            tracker_filenames = GetSlicedDownloadTrackerFilePaths(StorageUrlFromString(fpath), self.test_api)
            for tracker_filename in tracker_filenames:
                self.assertTrue(os.path.isfile(tracker_filename))
            stderr = self.RunGsUtil(['cp', suri(object_uri), fpath], return_stderr=True)
            self.assertIn('Resuming download', stderr)
            self.assertIn('Download already complete', stderr)
            tracker_filenames = GetSlicedDownloadTrackerFilePaths(StorageUrlFromString(fpath), self.test_api)
            for tracker_filename in tracker_filenames:
                self.assertFalse(os.path.isfile(tracker_filename))
            with open(fpath, 'rb') as f:
                self.assertEqual(f.read(), b'abc' * self.halt_size, 'File contents differ')

    @unittest.skipUnless(UsingCrcmodExtension(), 'Sliced download requires fast crcmod.')
    @SkipForS3('No sliced download support for S3.')
    def test_cp_sliced_download_resume_content_differs(self):
        """Tests differing file contents are detected by sliced downloads."""
        bucket_uri = self.CreateBucket()
        object_uri = self.CreateObject(bucket_uri=bucket_uri, object_name='foo', contents=b'abc' * self.halt_size)
        fpath = self.CreateTempFile(contents=b'')
        test_callback_file = self.CreateTempFile(contents=pickle.dumps(HaltingCopyCallbackHandler(False, 5)))
        boto_config_for_test = [('GSUtil', 'resumable_threshold', str(self.halt_size)), ('GSUtil', 'sliced_object_download_threshold', str(self.halt_size)), ('GSUtil', 'sliced_object_download_max_components', '3')]
        with SetBotoConfigForTest(boto_config_for_test):
            stderr = self.RunGsUtil(['cp', '--testcallbackfile', test_callback_file, suri(object_uri), suri(fpath)], return_stderr=True, expected_status=1)
            self.assertIn('not downloaded successfully', stderr)
            self.assertTrue(os.path.isfile(fpath + '_.gstmp'))
            tracker_filenames = GetSlicedDownloadTrackerFilePaths(StorageUrlFromString(fpath), self.test_api)
            for tracker_filename in tracker_filenames:
                self.assertTrue(os.path.isfile(tracker_filename))
            with open(fpath + '_.gstmp', 'r+b') as f:
                f.write(b'altered file contents')
            stderr = self.RunGsUtil(['cp', suri(object_uri), fpath], return_stderr=True, expected_status=1)
            self.assertIn('Resuming download', stderr)
            self.assertIn("doesn't match cloud-supplied digest", stderr)
            self.assertIn('HashMismatchException: crc32c', stderr)
            tracker_filenames = GetSlicedDownloadTrackerFilePaths(StorageUrlFromString(fpath), self.test_api)
            for tracker_filename in tracker_filenames:
                self.assertFalse(os.path.isfile(tracker_filename))
            self.assertFalse(os.path.isfile(fpath + '_.gstmp'))
            self.assertFalse(os.path.isfile(fpath))

    @unittest.skipUnless(UsingCrcmodExtension(), 'Sliced download requires fast crcmod.')
    @SkipForS3('No sliced download support for S3.')
    def test_cp_sliced_download_component_size_changed(self):
        """Tests sliced download doesn't break when the boto config changes.

    If the number of components used changes cross-process, the download should
    be restarted.
    """
        bucket_uri = self.CreateBucket()
        object_uri = self.CreateObject(bucket_uri=bucket_uri, object_name='foo', contents=b'abcd' * self.halt_size)
        fpath = self.CreateTempFile()
        test_callback_file = self.CreateTempFile(contents=pickle.dumps(HaltingCopyCallbackHandler(False, 5)))
        boto_config_for_test = [('GSUtil', 'resumable_threshold', str(self.halt_size)), ('GSUtil', 'sliced_object_download_threshold', str(self.halt_size)), ('GSUtil', 'sliced_object_download_component_size', str(self.halt_size // 4)), ('GSUtil', 'sliced_object_download_max_components', '4')]
        with SetBotoConfigForTest(boto_config_for_test):
            stderr = self.RunGsUtil(['cp', '--testcallbackfile', test_callback_file, suri(object_uri), suri(fpath)], return_stderr=True, expected_status=1)
            self.assertIn('not downloaded successfully', stderr)
        boto_config_for_test = [('GSUtil', 'resumable_threshold', str(self.halt_size)), ('GSUtil', 'sliced_object_download_threshold', str(self.halt_size)), ('GSUtil', 'sliced_object_download_component_size', str(self.halt_size // 2)), ('GSUtil', 'sliced_object_download_max_components', '2')]
        with SetBotoConfigForTest(boto_config_for_test):
            stderr = self.RunGsUtil(['cp', suri(object_uri), fpath], return_stderr=True)
            self.assertIn("Sliced download tracker file doesn't match ", stderr)
            self.assertIn('Restarting download from scratch', stderr)
            self.assertNotIn('Resuming download', stderr)

    @unittest.skipUnless(UsingCrcmodExtension(), 'Sliced download requires fast crcmod.')
    @SkipForS3('No sliced download support for S3.')
    def test_cp_sliced_download_disabled_cross_process(self):
        """Tests temporary files are not orphaned if sliced download is disabled.

    Specifically, temporary files should be deleted when the corresponding
    non-sliced download is completed.
    """
        bucket_uri = self.CreateBucket()
        object_uri = self.CreateObject(bucket_uri=bucket_uri, object_name='foo', contents=b'abcd' * self.halt_size)
        fpath = self.CreateTempFile()
        test_callback_file = self.CreateTempFile(contents=pickle.dumps(HaltingCopyCallbackHandler(False, 5)))
        boto_config_for_test = [('GSUtil', 'resumable_threshold', str(self.halt_size)), ('GSUtil', 'sliced_object_download_threshold', str(self.halt_size)), ('GSUtil', 'sliced_object_download_max_components', '4')]
        with SetBotoConfigForTest(boto_config_for_test):
            stderr = self.RunGsUtil(['cp', '--testcallbackfile', test_callback_file, suri(object_uri), suri(fpath)], return_stderr=True, expected_status=1)
            self.assertIn('not downloaded successfully', stderr)
            self.assertTrue(os.path.isfile(fpath + '_.gstmp'))
            tracker_filenames = GetSlicedDownloadTrackerFilePaths(StorageUrlFromString(fpath), self.test_api)
            for tracker_filename in tracker_filenames:
                self.assertTrue(os.path.isfile(tracker_filename))
        boto_config_for_test = [('GSUtil', 'resumable_threshold', str(self.halt_size)), ('GSUtil', 'sliced_object_download_threshold', str(self.halt_size * 5)), ('GSUtil', 'sliced_object_download_max_components', '4')]
        with SetBotoConfigForTest(boto_config_for_test):
            stderr = self.RunGsUtil(['cp', suri(object_uri), fpath], return_stderr=True)
            self.assertNotIn('Resuming download', stderr)
            self.assertFalse(os.path.isfile(fpath + '_.gstmp'))
            for tracker_filename in tracker_filenames:
                self.assertFalse(os.path.isfile(tracker_filename))
            with open(fpath, 'rb') as f:
                self.assertEqual(f.read(), b'abcd' * self.halt_size)

    @SkipForS3('No resumable upload support for S3.')
    def test_cp_resumable_upload_start_over_http_error(self):
        for start_over_error in (403, 404, 410):
            self.start_over_error_test_helper(start_over_error)

    def start_over_error_test_helper(self, http_error_num):
        bucket_uri = self.CreateBucket()
        rand_chars = get_random_ascii_chars(size=ONE_MIB * 4)
        fpath = self.CreateTempFile(contents=rand_chars)
        boto_config_for_test = ('GSUtil', 'resumable_threshold', str(ONE_KIB))
        if self.test_api == ApiSelector.JSON:
            test_callback_file = self.CreateTempFile(contents=pickle.dumps(_JSONForceHTTPErrorCopyCallbackHandler(5, 404)))
        elif self.test_api == ApiSelector.XML:
            test_callback_file = self.CreateTempFile(contents=pickle.dumps(_XMLResumableUploadStartOverCopyCallbackHandler(5)))
        with SetBotoConfigForTest([boto_config_for_test]):
            stderr = self.RunGsUtil(['cp', '--testcallbackfile', test_callback_file, fpath, suri(bucket_uri)], return_stderr=True)
            self.assertIn('Restarting upload of', stderr)

    def test_cp_minus_c(self):
        bucket_uri = self.CreateBucket()
        object_uri = self.CreateObject(bucket_uri=bucket_uri, object_name='foo', contents=b'foo')
        cp_command = ['cp', '-c', suri(bucket_uri) + '/foo2', suri(object_uri), suri(bucket_uri) + '/dir/']
        self.RunGsUtil(cp_command, expected_status=1)
        self.RunGsUtil(['stat', '%s/dir/foo' % suri(bucket_uri)])

    def test_rewrite_cp(self):
        """Tests the JSON Rewrite API."""
        if self.test_api == ApiSelector.XML:
            return unittest.skip('Rewrite API is only supported in JSON.')
        bucket_uri = self.CreateBucket()
        object_uri = self.CreateObject(bucket_uri=bucket_uri, object_name='foo', contents=b'bar')
        gsutil_api = GcsJsonApi(BucketStorageUri, logging.getLogger(), DiscardMessagesQueue(), self.default_provider)
        key = object_uri.get_key()
        src_obj_metadata = apitools_messages.Object(name=key.name, bucket=key.bucket.name, contentType=key.content_type)
        dst_obj_metadata = apitools_messages.Object(bucket=src_obj_metadata.bucket, name=self.MakeTempName('object'), contentType=src_obj_metadata.contentType)
        gsutil_api.CopyObject(src_obj_metadata, dst_obj_metadata)
        self.assertEqual(gsutil_api.GetObjectMetadata(src_obj_metadata.bucket, src_obj_metadata.name, fields=['customerEncryption', 'md5Hash']).md5Hash, gsutil_api.GetObjectMetadata(dst_obj_metadata.bucket, dst_obj_metadata.name, fields=['customerEncryption', 'md5Hash']).md5Hash, "Error: Rewritten object's hash doesn't match source object.")

    def test_rewrite_cp_resume(self):
        """Tests the JSON Rewrite API, breaking and resuming via a tracker file."""
        if self.test_api == ApiSelector.XML:
            return unittest.skip('Rewrite API is only supported in JSON.')
        bucket_uri = self.CreateBucket()
        bucket_uri2 = self.CreateBucket(storage_class='durable_reduced_availability')
        object_uri = self.CreateObject(bucket_uri=bucket_uri, object_name='foo', contents=b'12' * ONE_MIB + b'bar', prefer_json_api=True)
        gsutil_api = GcsJsonApi(BucketStorageUri, logging.getLogger(), DiscardMessagesQueue(), self.default_provider)
        key = object_uri.get_key()
        src_obj_metadata = apitools_messages.Object(name=key.name, bucket=key.bucket.name, contentType=key.content_type, etag=key.etag.strip('"\''))
        dst_obj_name = self.MakeTempName('object')
        dst_obj_metadata = apitools_messages.Object(bucket=bucket_uri2.bucket_name, name=dst_obj_name, contentType=src_obj_metadata.contentType)
        tracker_file_name = GetRewriteTrackerFilePath(src_obj_metadata.bucket, src_obj_metadata.name, dst_obj_metadata.bucket, dst_obj_metadata.name, self.test_api)
        try:
            try:
                gsutil_api.CopyObject(src_obj_metadata, dst_obj_metadata, progress_callback=HaltingRewriteCallbackHandler(ONE_MIB * 2).call, max_bytes_per_call=ONE_MIB)
                self.fail('Expected RewriteHaltException.')
            except RewriteHaltException:
                pass
            self.assertTrue(os.path.exists(tracker_file_name))
            gsutil_api.CopyObject(src_obj_metadata, dst_obj_metadata, progress_callback=EnsureRewriteResumeCallbackHandler(ONE_MIB * 2).call, max_bytes_per_call=ONE_MIB)
            self.assertFalse(os.path.exists(tracker_file_name))
            self.assertEqual(gsutil_api.GetObjectMetadata(src_obj_metadata.bucket, src_obj_metadata.name, fields=['customerEncryption', 'md5Hash']).md5Hash, gsutil_api.GetObjectMetadata(dst_obj_metadata.bucket, dst_obj_metadata.name, fields=['customerEncryption', 'md5Hash']).md5Hash, "Error: Rewritten object's hash doesn't match source object.")
        finally:
            DeleteTrackerFile(tracker_file_name)

    def test_rewrite_cp_resume_source_changed(self):
        """Tests that Rewrite starts over when the source object has changed."""
        if self.test_api == ApiSelector.XML:
            return unittest.skip('Rewrite API is only supported in JSON.')
        bucket_uri = self.CreateBucket()
        bucket_uri2 = self.CreateBucket(storage_class='durable_reduced_availability')
        object_uri = self.CreateObject(bucket_uri=bucket_uri, object_name='foo', contents=b'12' * ONE_MIB + b'bar', prefer_json_api=True)
        gsutil_api = GcsJsonApi(BucketStorageUri, logging.getLogger(), DiscardMessagesQueue(), self.default_provider)
        key = object_uri.get_key()
        src_obj_metadata = apitools_messages.Object(name=key.name, bucket=key.bucket.name, contentType=key.content_type, etag=key.etag.strip('"\''))
        dst_obj_name = self.MakeTempName('object')
        dst_obj_metadata = apitools_messages.Object(bucket=bucket_uri2.bucket_name, name=dst_obj_name, contentType=src_obj_metadata.contentType)
        tracker_file_name = GetRewriteTrackerFilePath(src_obj_metadata.bucket, src_obj_metadata.name, dst_obj_metadata.bucket, dst_obj_metadata.name, self.test_api)
        try:
            try:
                gsutil_api.CopyObject(src_obj_metadata, dst_obj_metadata, progress_callback=HaltingRewriteCallbackHandler(ONE_MIB * 2).call, max_bytes_per_call=ONE_MIB)
                self.fail('Expected RewriteHaltException.')
            except RewriteHaltException:
                pass
            object_uri2 = self.CreateObject(bucket_uri=bucket_uri, object_name='foo', contents=b'bar', prefer_json_api=True)
            key2 = object_uri2.get_key()
            src_obj_metadata2 = apitools_messages.Object(name=key2.name, bucket=key2.bucket.name, contentType=key2.content_type, etag=key2.etag.strip('"\''))
            self.assertTrue(os.path.exists(tracker_file_name))
            gsutil_api.CopyObject(src_obj_metadata2, dst_obj_metadata, max_bytes_per_call=ONE_MIB)
            self.assertFalse(os.path.exists(tracker_file_name))
            self.assertEqual(gsutil_api.GetObjectMetadata(src_obj_metadata2.bucket, src_obj_metadata2.name, fields=['customerEncryption', 'md5Hash']).md5Hash, gsutil_api.GetObjectMetadata(dst_obj_metadata.bucket, dst_obj_metadata.name, fields=['customerEncryption', 'md5Hash']).md5Hash, "Error: Rewritten object's hash doesn't match source object.")
        finally:
            DeleteTrackerFile(tracker_file_name)

    def test_rewrite_cp_resume_command_changed(self):
        """Tests that Rewrite starts over when the arguments changed."""
        if self.test_api == ApiSelector.XML:
            return unittest.skip('Rewrite API is only supported in JSON.')
        bucket_uri = self.CreateBucket()
        bucket_uri2 = self.CreateBucket(storage_class='durable_reduced_availability')
        object_uri = self.CreateObject(bucket_uri=bucket_uri, object_name='foo', contents=b'12' * ONE_MIB + b'bar', prefer_json_api=True)
        gsutil_api = GcsJsonApi(BucketStorageUri, logging.getLogger(), DiscardMessagesQueue(), self.default_provider)
        key = object_uri.get_key()
        src_obj_metadata = apitools_messages.Object(name=key.name, bucket=key.bucket.name, contentType=key.content_type, etag=key.etag.strip('"\''))
        dst_obj_name = self.MakeTempName('object')
        dst_obj_metadata = apitools_messages.Object(bucket=bucket_uri2.bucket_name, name=dst_obj_name, contentType=src_obj_metadata.contentType)
        tracker_file_name = GetRewriteTrackerFilePath(src_obj_metadata.bucket, src_obj_metadata.name, dst_obj_metadata.bucket, dst_obj_metadata.name, self.test_api)
        try:
            try:
                gsutil_api.CopyObject(src_obj_metadata, dst_obj_metadata, canned_acl='private', progress_callback=HaltingRewriteCallbackHandler(ONE_MIB * 2).call, max_bytes_per_call=ONE_MIB)
                self.fail('Expected RewriteHaltException.')
            except RewriteHaltException:
                pass
            self.assertTrue(os.path.exists(tracker_file_name))
            gsutil_api.CopyObject(src_obj_metadata, dst_obj_metadata, canned_acl='public-read', max_bytes_per_call=ONE_MIB)
            self.assertFalse(os.path.exists(tracker_file_name))
            new_obj_metadata = gsutil_api.GetObjectMetadata(dst_obj_metadata.bucket, dst_obj_metadata.name, fields=['acl', 'customerEncryption', 'md5Hash'])
            self.assertEqual(gsutil_api.GetObjectMetadata(src_obj_metadata.bucket, src_obj_metadata.name, fields=['customerEncryption', 'md5Hash']).md5Hash, new_obj_metadata.md5Hash, "Error: Rewritten object's hash doesn't match source object.")
            found_public_acl = False
            for acl_entry in new_obj_metadata.acl:
                if acl_entry.entity == 'allUsers':
                    found_public_acl = True
            self.assertTrue(found_public_acl, 'New object was not written with a public ACL.')
        finally:
            DeleteTrackerFile(tracker_file_name)

    @unittest.skipIf(IS_WINDOWS, 'POSIX attributes not available on Windows.')
    @unittest.skipUnless(UsingCrcmodExtension(), 'Test requires fast crcmod.')
    def test_cp_preserve_posix_bucket_to_dir_no_errors(self):
        """Tests use of the -P flag with cp from a bucket to a local dir.

    Specifically tests combinations of POSIX attributes in metadata that will
    pass validation.
    """
        bucket_uri = self.CreateBucket()
        tmpdir = self.CreateTempDir()
        TestCpMvPOSIXBucketToLocalNoErrors(self, bucket_uri, tmpdir, is_cp=True)

    @unittest.skipIf(IS_WINDOWS, 'POSIX attributes not available on Windows.')
    def test_cp_preserve_posix_bucket_to_dir_errors(self):
        """Tests use of the -P flag with cp from a bucket to a local dir.

    Specifically, combinations of POSIX attributes in metadata that will fail
    validation.
    """
        bucket_uri = self.CreateBucket()
        tmpdir = self.CreateTempDir()
        obj = self.CreateObject(bucket_uri=bucket_uri, object_name='obj', contents=b'obj')
        TestCpMvPOSIXBucketToLocalErrors(self, bucket_uri, obj, tmpdir, is_cp=True)

    @unittest.skipIf(IS_WINDOWS, 'POSIX attributes not available on Windows.')
    def test_cp_preseve_posix_dir_to_bucket_no_errors(self):
        """Tests use of the -P flag with cp from a local dir to a bucket."""
        bucket_uri = self.CreateBucket()
        TestCpMvPOSIXLocalToBucketNoErrors(self, bucket_uri, is_cp=True)

    def test_cp_minus_s_to_non_cloud_dest_fails(self):
        """Test that cp -s operations to a non-cloud destination are prevented."""
        local_file = self.CreateTempFile(contents=b'foo')
        dest_dir = self.CreateTempDir()
        stderr = self.RunGsUtil(['cp', '-s', 'standard', local_file, dest_dir], expected_status=1, return_stderr=True)
        self.assertIn('Cannot specify storage class for a non-cloud destination:', stderr)

    @SkipForXML('Need Boto version > 2.46.1')
    def test_cp_specify_nondefault_storage_class(self):
        bucket_uri = self.CreateBucket()
        object_uri = self.CreateObject(bucket_uri=bucket_uri, object_name='foo', contents=b'foo')
        object2_suri = suri(object_uri) + 'bar'
        nondefault_storage_class = {'s3': 'Standard_iA', 'gs': 'durable_REDUCED_availability'}
        storage_class = nondefault_storage_class[self.default_provider]
        self.RunGsUtil(['cp', '-s', storage_class, suri(object_uri), object2_suri])
        stdout = self.RunGsUtil(['stat', object2_suri], return_stdout=True)
        self.assertRegexpMatchesWithFlags(stdout, 'Storage class:\\s+%s' % storage_class, flags=re.IGNORECASE)

    @SkipForS3('Test uses gs-specific storage classes.')
    def test_cp_sets_correct_dest_storage_class(self):
        """Tests that object storage class is set correctly with and without -s."""
        bucket_uri = self.CreateBucket(storage_class='nearline')
        local_fname = 'foo-orig'
        local_fpath = self.CreateTempFile(contents=b'foo', file_name=local_fname)
        foo_cloud_suri = suri(bucket_uri) + '/' + local_fname
        self.RunGsUtil(['cp', '-s', 'standard', local_fpath, foo_cloud_suri])
        with SetBotoConfigForTest([('GSUtil', 'prefer_api', 'json')]):
            stdout = self.RunGsUtil(['stat', foo_cloud_suri], return_stdout=True)
        self.assertRegexpMatchesWithFlags(stdout, 'Storage class:\\s+STANDARD', flags=re.IGNORECASE)
        foo_nl_suri = suri(bucket_uri) + '/foo-nl'
        self.RunGsUtil(['cp', foo_cloud_suri, foo_nl_suri])
        with SetBotoConfigForTest([('GSUtil', 'prefer_api', 'json')]):
            stdout = self.RunGsUtil(['stat', foo_nl_suri], return_stdout=True)
        self.assertRegexpMatchesWithFlags(stdout, 'Storage class:\\s+NEARLINE', flags=re.IGNORECASE)
        foo_std_suri = suri(bucket_uri) + '/foo-std'
        self.RunGsUtil(['cp', '-s', 'standard', foo_nl_suri, foo_std_suri])
        with SetBotoConfigForTest([('GSUtil', 'prefer_api', 'json')]):
            stdout = self.RunGsUtil(['stat', foo_std_suri], return_stdout=True)
        self.assertRegexpMatchesWithFlags(stdout, 'Storage class:\\s+STANDARD', flags=re.IGNORECASE)

    @SkipForS3('Test uses gs-specific KMS encryption')
    def test_kms_key_correctly_applied_to_dst_obj_from_src_with_no_key(self):
        bucket_uri = self.CreateBucket()
        obj1_name = 'foo'
        obj2_name = 'bar'
        key_fqn = AuthorizeProjectToUseTestingKmsKey()
        obj_uri = self.CreateObject(bucket_uri=bucket_uri, object_name=obj1_name, contents=b'foo')
        with SetBotoConfigForTest([('GSUtil', 'encryption_key', key_fqn)]):
            self.RunGsUtil(['cp', suri(obj_uri), '%s/%s' % (suri(bucket_uri), obj2_name)])
        with SetBotoConfigForTest([('GSUtil', 'prefer_api', 'json')]):
            self.AssertObjectUsesCMEK('%s/%s' % (suri(bucket_uri), obj2_name), key_fqn)

    @SkipForS3('Test uses gs-specific KMS encryption')
    def test_kms_key_correctly_applied_to_dst_obj_from_local_file(self):
        bucket_uri = self.CreateBucket()
        fpath = self.CreateTempFile(contents=b'abcd')
        obj_name = 'foo'
        obj_suri = suri(bucket_uri) + '/' + obj_name
        key_fqn = AuthorizeProjectToUseTestingKmsKey()
        with SetBotoConfigForTest([('GSUtil', 'encryption_key', key_fqn)]):
            self.RunGsUtil(['cp', fpath, obj_suri])
        with SetBotoConfigForTest([('GSUtil', 'prefer_api', 'json')]):
            self.AssertObjectUsesCMEK(obj_suri, key_fqn)

    @SkipForS3('Test uses gs-specific KMS encryption')
    def test_kms_key_works_with_resumable_upload(self):
        resumable_threshold = 1024 * 1024
        bucket_uri = self.CreateBucket()
        fpath = self.CreateTempFile(contents=b'a' * resumable_threshold)
        obj_name = 'foo'
        obj_suri = suri(bucket_uri) + '/' + obj_name
        key_fqn = AuthorizeProjectToUseTestingKmsKey()
        with SetBotoConfigForTest([('GSUtil', 'encryption_key', key_fqn), ('GSUtil', 'resumable_threshold', str(resumable_threshold))]):
            self.RunGsUtil(['cp', fpath, obj_suri])
        with SetBotoConfigForTest([('GSUtil', 'prefer_api', 'json')]):
            self.AssertObjectUsesCMEK(obj_suri, key_fqn)

    @SkipForS3('Test uses gs-specific KMS encryption')
    def test_kms_key_correctly_applied_to_dst_obj_from_src_with_diff_key(self):
        bucket_uri = self.CreateBucket()
        obj1_name = 'foo'
        obj2_name = 'bar'
        key1_fqn = AuthorizeProjectToUseTestingKmsKey()
        key2_fqn = AuthorizeProjectToUseTestingKmsKey(key_name=KmsTestingResources.CONSTANT_KEY_NAME2)
        obj1_suri = suri(self.CreateObject(bucket_uri=bucket_uri, object_name=obj1_name, contents=b'foo', kms_key_name=key1_fqn))
        obj2_suri = '%s/%s' % (suri(bucket_uri), obj2_name)
        with SetBotoConfigForTest([('GSUtil', 'encryption_key', key2_fqn)]):
            self.RunGsUtil(['cp', obj1_suri, obj2_suri])
        with SetBotoConfigForTest([('GSUtil', 'prefer_api', 'json')]):
            self.AssertObjectUsesCMEK(obj2_suri, key2_fqn)

    @SkipForS3('Test uses gs-specific KMS encryption')
    @SkipForXML('Copying KMS-encrypted objects prohibited with XML API')
    def test_kms_key_not_applied_to_nonkms_dst_obj_from_src_with_kms_key(self):
        bucket_uri = self.CreateBucket()
        obj1_name = 'foo'
        obj2_name = 'bar'
        key1_fqn = AuthorizeProjectToUseTestingKmsKey()
        obj1_suri = suri(self.CreateObject(bucket_uri=bucket_uri, object_name=obj1_name, contents=b'foo', kms_key_name=key1_fqn))
        obj2_suri = '%s/%s' % (suri(bucket_uri), obj2_name)
        self.RunGsUtil(['cp', obj1_suri, obj2_suri])
        with SetBotoConfigForTest([('GSUtil', 'prefer_api', 'json')]):
            self.AssertObjectUnencrypted(obj2_suri)

    @unittest.skipUnless(IS_WINDOWS, 'Only Windows paths need to be normalized to use backslashes instead of forward slashes.')
    def test_windows_path_with_back_and_forward_slash_is_normalized(self):
        tmp_dir = self.CreateTempDir()
        self.CreateTempFile(tmpdir=tmp_dir, file_name='obj1', contents=b'foo')
        bucket_uri = self.CreateBucket()
        self.RunGsUtil(['cp', '%s\\./obj1' % tmp_dir, suri(bucket_uri)])
        self.RunGsUtil(['stat', '%s/obj1' % suri(bucket_uri)])

    def test_cp_minus_m_streaming_upload(self):
        """Tests that cp -m - anything is disallowed."""
        stderr = self.RunGsUtil(['-m', 'cp', '-', 'file'], return_stderr=True, expected_status=1)
        if self._use_gcloud_storage:
            self.assertIn('WARNING: Using sequential instead of parallel task execution to transfer from stdin', stderr)
        else:
            self.assertIn('CommandException: Cannot upload from a stream when using gsutil -m', stderr)

    @SequentialAndParallelTransfer
    def test_cp_overwrites_existing_destination(self):
        key_uri = self.CreateObject(contents=b'foo')
        fpath = self.CreateTempFile(contents=b'bar')
        stderr = self.RunGsUtil(['cp', suri(key_uri), fpath], return_stderr=True)
        with open(fpath, 'rb') as f:
            self.assertEqual(f.read(), b'foo')

    @SequentialAndParallelTransfer
    def test_downloads_are_reliable_with_more_than_one_gsutil_instance(self):
        test_file_count = 10
        temporary_directory = self.CreateTempDir()
        bucket_uri = self.CreateBucket(test_objects=test_file_count)
        cp_args = ['cp', suri(bucket_uri, '*'), temporary_directory]
        threads = []
        for _ in range(2):
            thread = threading.Thread(target=self.RunGsUtil, args=[cp_args])
            thread.start()
            threads.append(thread)
        [t.join() for t in threads]
        self.assertEqual(len(os.listdir(temporary_directory)), test_file_count)