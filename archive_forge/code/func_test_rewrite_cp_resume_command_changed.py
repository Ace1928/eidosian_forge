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