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
def TestCpMvPOSIXBucketToLocalNoErrors(cls, bucket_uri, tmpdir, is_cp=True):
    """Helper function for preserve_posix_no_errors tests in test_cp and test_mv.

  Args:
    cls: An instance of either TestCp or TestMv.
    bucket_uri: The uri of the bucket that the object is in.
    tmpdir: The local file path to cp to.
    is_cp: Whether or not the calling test suite is cp or mv.
  """
    primary_gid = os.stat(tmpdir).st_gid
    non_primary_gid = util.GetNonPrimaryGid()
    test_params = {'obj1': {GID_ATTR: primary_gid}, 'obj2': {GID_ATTR: non_primary_gid}, 'obj3': {GID_ATTR: primary_gid, MODE_ATTR: '440'}, 'obj4': {GID_ATTR: non_primary_gid, MODE_ATTR: '444'}, 'obj5': {UID_ATTR: USER_ID}, 'obj6': {UID_ATTR: USER_ID, MODE_ATTR: '420'}, 'obj7': {UID_ATTR: USER_ID, GID_ATTR: primary_gid}, 'obj8': {UID_ATTR: USER_ID, GID_ATTR: non_primary_gid}, 'obj9': {UID_ATTR: USER_ID, GID_ATTR: primary_gid, MODE_ATTR: '433'}, 'obj10': {UID_ATTR: USER_ID, GID_ATTR: non_primary_gid, MODE_ATTR: '442'}}
    for obj_name, attrs_dict in six.iteritems(test_params):
        uid = attrs_dict.get(UID_ATTR)
        gid = attrs_dict.get(GID_ATTR)
        mode = attrs_dict.get(MODE_ATTR)
        cls.CreateObject(bucket_uri=bucket_uri, object_name=obj_name, contents=obj_name.encode(UTF8), uid=uid, gid=gid, mode=mode)
    for obj_name in six.iterkeys(test_params):
        cls.RunGsUtil(['cp' if is_cp else 'mv', '-P', suri(bucket_uri, obj_name), tmpdir])
    listing = TailSet(tmpdir, cls.FlatListDir(tmpdir))
    cls.assertEqual(listing, set(['/obj1', '/obj2', '/obj3', '/obj4', '/obj5', '/obj6', '/obj7', '/obj8', '/obj9', '/obj10']))
    cls.VerifyLocalPOSIXPermissions(os.path.join(tmpdir, 'obj1'), gid=primary_gid, mode=DEFAULT_MODE)
    cls.VerifyLocalPOSIXPermissions(os.path.join(tmpdir, 'obj2'), gid=non_primary_gid, mode=DEFAULT_MODE)
    cls.VerifyLocalPOSIXPermissions(os.path.join(tmpdir, 'obj3'), gid=primary_gid, mode=288)
    cls.VerifyLocalPOSIXPermissions(os.path.join(tmpdir, 'obj4'), gid=non_primary_gid, mode=292)
    cls.VerifyLocalPOSIXPermissions(os.path.join(tmpdir, 'obj5'), uid=USER_ID, gid=primary_gid, mode=DEFAULT_MODE)
    cls.VerifyLocalPOSIXPermissions(os.path.join(tmpdir, 'obj6'), uid=USER_ID, gid=primary_gid, mode=272)
    cls.VerifyLocalPOSIXPermissions(os.path.join(tmpdir, 'obj7'), uid=USER_ID, gid=primary_gid, mode=DEFAULT_MODE)
    cls.VerifyLocalPOSIXPermissions(os.path.join(tmpdir, 'obj8'), uid=USER_ID, gid=non_primary_gid, mode=DEFAULT_MODE)
    cls.VerifyLocalPOSIXPermissions(os.path.join(tmpdir, 'obj9'), uid=USER_ID, gid=primary_gid, mode=283)
    cls.VerifyLocalPOSIXPermissions(os.path.join(tmpdir, 'obj10'), uid=USER_ID, gid=non_primary_gid, mode=290)