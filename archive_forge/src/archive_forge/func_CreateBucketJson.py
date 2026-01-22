from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import contextlib
import datetime
import locale
import logging
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
import boto
from boto import config
from boto.exception import StorageResponseError
from boto.s3.deletemarker import DeleteMarker
from boto.storage_uri import BucketStorageUri
import gslib
from gslib.boto_translation import BotoTranslation
from gslib.cloud_api import PreconditionException
from gslib.cloud_api import Preconditions
from gslib.discard_messages_queue import DiscardMessagesQueue
from gslib.exception import CommandException
from gslib.gcs_json_api import GcsJsonApi
from gslib.kms_api import KmsApi
from gslib.project_id import GOOG_PROJ_ID_HDR
from gslib.project_id import PopulateProjectId
from gslib.tests.testcase import base
import gslib.tests.util as util
from gslib.tests.util import InvokedFromParFile
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import RUN_S3_TESTS
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import SetEnvironmentForTest
from gslib.tests.util import unittest
from gslib.tests.util import USING_JSON_API
import gslib.third_party.storage_apitools.storage_v1_messages as apitools_messages
from gslib.utils.constants import UTF8
from gslib.utils.encryption_helper import Base64Sha256FromBase64EncryptionKey
from gslib.utils.encryption_helper import CryptoKeyWrapperFromKey
from gslib.utils.hashing_helper import Base64ToHexHash
from gslib.utils.metadata_util import CreateCustomMetadata
from gslib.utils.metadata_util import GetValueFromObjectCustomMetadata
from gslib.utils.posix_util import ATIME_ATTR
from gslib.utils.posix_util import GID_ATTR
from gslib.utils.posix_util import MODE_ATTR
from gslib.utils.posix_util import MTIME_ATTR
from gslib.utils.posix_util import UID_ATTR
from gslib.utils.retry_util import Retry
import six
from six.moves import range
def CreateBucketJson(self, bucket_name=None, test_objects=0, storage_class=None, location=None, versioning_enabled=False, retention_policy=None, bucket_policy_only=False, public_access_prevention=None):
    """Creates a test bucket using the JSON API.

    The bucket and all of its contents will be deleted after the test.

    Args:
      bucket_name: Create the bucket with this name. If not provided, a
                   temporary test bucket name is constructed.
      test_objects: The number of objects that should be placed in the bucket.
                    Defaults to 0.
      storage_class: Storage class to use. If not provided we use standard.
      location: Location to use.
      versioning_enabled: If True, set the bucket's versioning attribute to
          True.
      retention_policy: Retention policy to be used on the bucket.
      bucket_policy_only: If True, set the bucket's iamConfiguration's
          bucketPolicyOnly attribute to True.
      public_access_prevention: String value of public access prevention. Valid
          values are "enforced" and "unspecified".

    Returns:
      Apitools Bucket for the created bucket.
    """
    bucket_name = util.MakeBucketNameValid(bucket_name or self.MakeTempName('bucket'))
    bucket_metadata = apitools_messages.Bucket(name=bucket_name.lower())
    if storage_class:
        bucket_metadata.storageClass = storage_class
    if location:
        bucket_metadata.location = location
    if versioning_enabled:
        bucket_metadata.versioning = apitools_messages.Bucket.VersioningValue(enabled=True)
    if retention_policy:
        bucket_metadata.retentionPolicy = retention_policy
    if bucket_policy_only or public_access_prevention:
        iam_config = apitools_messages.Bucket.IamConfigurationValue()
        if bucket_policy_only:
            iam_config.bucketPolicyOnly = iam_config.BucketPolicyOnlyValue()
            iam_config.bucketPolicyOnly.enabled = True
        if public_access_prevention:
            iam_config.publicAccessPrevention = public_access_prevention
        bucket_metadata.iamConfiguration = iam_config
    bucket = self.json_api.CreateBucket(bucket_name, metadata=bucket_metadata)
    self.bucket_uris.append(boto.storage_uri('gs://%s' % bucket_name, suppress_consec_slashes=False))
    for i in range(test_objects):
        self.CreateObjectJson(bucket_name=bucket_name, object_name=self.MakeTempName('obj'), contents='test {:d}'.format(i).encode('ascii'))
    return bucket