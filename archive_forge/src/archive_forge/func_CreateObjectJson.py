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
def CreateObjectJson(self, contents, bucket_name=None, object_name=None, encryption_key=None, mtime=None, storage_class=None, gs_idempotent_generation=None, kms_key_name=None):
    """Creates a test object (GCS provider only) using the JSON API.

    Args:
      contents: The contents to write to the object.
      bucket_name: Name of bucket to place the object in. If not specified,
          a new temporary bucket is created. Assumes the given bucket name is
          valid.
      object_name: The name to use for the object. If not specified, a temporary
          test object name is constructed.
      encryption_key: AES256 encryption key to use when creating the object,
          if any.
      mtime: The modification time of the file in POSIX time (seconds since
          UTC 1970-01-01). If not specified, this defaults to the current
          system time.
      storage_class: String representing the storage class to use for the
          object.
      gs_idempotent_generation: For use when overwriting an object for which
          you know the previously uploaded generation. Create GCS object
          idempotently by supplying this generation number as a precondition
          and assuming the current object is correct on precondition failure.
          Defaults to 0 (new object); to disable, set to None.
      kms_key_name: Fully-qualified name of the KMS key that should be used to
          encrypt the object. Note that this is currently only valid for 'gs'
          objects.

    Returns:
      An apitools Object for the created object.
    """
    bucket_name = bucket_name or self.CreateBucketJson().name
    object_name = object_name or self.MakeTempName('obj')
    preconditions = Preconditions(gen_match=gs_idempotent_generation)
    custom_metadata = apitools_messages.Object.MetadataValue(additionalProperties=[])
    if mtime is not None:
        CreateCustomMetadata({MTIME_ATTR: mtime}, custom_metadata)
    object_metadata = apitools_messages.Object(name=object_name, metadata=custom_metadata, bucket=bucket_name, contentType='application/octet-stream', storageClass=storage_class, kmsKeyName=kms_key_name)
    encryption_keywrapper = CryptoKeyWrapperFromKey(encryption_key)
    try:
        return self.json_api.UploadObject(six.BytesIO(contents), object_metadata, provider='gs', encryption_tuple=encryption_keywrapper, preconditions=preconditions)
    except PreconditionException:
        if gs_idempotent_generation is None:
            raise
        with SetBotoConfigForTest([('GSUtil', 'decryption_key1', encryption_key)]):
            return self.json_api.GetObjectMetadata(bucket_name, object_name)