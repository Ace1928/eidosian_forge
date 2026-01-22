from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import base64
from collections import namedtuple
import csv
import datetime
import errno
import gzip
import json
import logging
import mimetypes
from operator import attrgetter
import os
import pickle
import pyu2f
import random
import re
import shutil
import six
import stat
import subprocess
import tempfile
import textwrap
import time
import traceback
import six
from six.moves import xrange
from six.moves import range
from apitools.base.protorpclite import protojson
from boto import config
import crcmod
import gslib
from gslib.cloud_api import AccessDeniedException
from gslib.cloud_api import ArgumentException
from gslib.cloud_api import CloudApi
from gslib.cloud_api import EncryptionException
from gslib.cloud_api import NotFoundException
from gslib.cloud_api import PreconditionException
from gslib.cloud_api import Preconditions
from gslib.cloud_api import ResumableDownloadException
from gslib.cloud_api import ResumableUploadAbortException
from gslib.cloud_api import ResumableUploadException
from gslib.cloud_api import ResumableUploadStartOverException
from gslib.cloud_api import ServiceException
from gslib.commands.compose import MAX_COMPOSE_ARITY
from gslib.commands.config import DEFAULT_PARALLEL_COMPOSITE_UPLOAD_COMPONENT_SIZE
from gslib.commands.config import DEFAULT_PARALLEL_COMPOSITE_UPLOAD_THRESHOLD
from gslib.commands.config import DEFAULT_SLICED_OBJECT_DOWNLOAD_COMPONENT_SIZE
from gslib.commands.config import DEFAULT_SLICED_OBJECT_DOWNLOAD_MAX_COMPONENTS
from gslib.commands.config import DEFAULT_SLICED_OBJECT_DOWNLOAD_THRESHOLD
from gslib.commands.config import DEFAULT_GZIP_COMPRESSION_LEVEL
from gslib.cs_api_map import ApiSelector
from gslib.daisy_chain_wrapper import DaisyChainWrapper
from gslib.exception import CommandException
from gslib.exception import HashMismatchException
from gslib.exception import InvalidUrlError
from gslib.file_part import FilePart
from gslib.parallel_tracker_file import GenerateComponentObjectPrefix
from gslib.parallel_tracker_file import ReadParallelUploadTrackerFile
from gslib.parallel_tracker_file import ValidateParallelCompositeTrackerData
from gslib.parallel_tracker_file import WriteComponentToParallelUploadTrackerFile
from gslib.parallel_tracker_file import WriteParallelUploadTrackerFile
from gslib.progress_callback import FileProgressCallbackHandler
from gslib.progress_callback import ProgressCallbackWithTimeout
from gslib.resumable_streaming_upload import ResumableStreamingJsonUploadWrapper
from gslib import storage_url
from gslib.storage_url import ContainsWildcard
from gslib.storage_url import GenerationFromUrlAndString
from gslib.storage_url import IsCloudSubdirPlaceholder
from gslib.storage_url import StorageUrlFromString
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
from gslib.thread_message import FileMessage
from gslib.thread_message import RetryableErrorMessage
from gslib.tracker_file import DeleteDownloadTrackerFiles
from gslib.tracker_file import DeleteTrackerFile
from gslib.tracker_file import ENCRYPTION_UPLOAD_TRACKER_ENTRY
from gslib.tracker_file import GetDownloadStartByte
from gslib.tracker_file import GetTrackerFilePath
from gslib.tracker_file import GetUploadTrackerData
from gslib.tracker_file import RaiseUnwritableTrackerFileException
from gslib.tracker_file import ReadOrCreateDownloadTrackerFile
from gslib.tracker_file import SERIALIZATION_UPLOAD_TRACKER_ENTRY
from gslib.tracker_file import TrackerFileType
from gslib.tracker_file import WriteDownloadComponentTrackerFile
from gslib.tracker_file import WriteJsonDataToTrackerFile
from gslib.utils import parallelism_framework_util
from gslib.utils import stet_util
from gslib.utils import temporary_file_util
from gslib.utils import text_util
from gslib.utils.boto_util import GetJsonResumableChunkSize
from gslib.utils.boto_util import GetMaxRetryDelay
from gslib.utils.boto_util import GetNumRetries
from gslib.utils.boto_util import ResumableThreshold
from gslib.utils.boto_util import UsingCrcmodExtension
from gslib.utils.cloud_api_helper import GetCloudApiInstance
from gslib.utils.cloud_api_helper import GetDownloadSerializationData
from gslib.utils.constants import DEFAULT_FILE_BUFFER_SIZE
from gslib.utils.constants import MIN_SIZE_COMPUTE_LOGGING
from gslib.utils.constants import UTF8
from gslib.utils.encryption_helper import CryptoKeyType
from gslib.utils.encryption_helper import CryptoKeyWrapperFromKey
from gslib.utils.encryption_helper import FindMatchingCSEKInBotoConfig
from gslib.utils.encryption_helper import GetEncryptionKeyWrapper
from gslib.utils.hashing_helper import Base64EncodeHash
from gslib.utils.hashing_helper import CalculateB64EncodedMd5FromContents
from gslib.utils.hashing_helper import CalculateHashesFromContents
from gslib.utils.hashing_helper import CHECK_HASH_IF_FAST_ELSE_FAIL
from gslib.utils.hashing_helper import CHECK_HASH_NEVER
from gslib.utils.hashing_helper import ConcatCrc32c
from gslib.utils.hashing_helper import GetDownloadHashAlgs
from gslib.utils.hashing_helper import GetMd5
from gslib.utils.hashing_helper import GetUploadHashAlgs
from gslib.utils.hashing_helper import HashingFileUploadWrapper
from gslib.utils.metadata_util import ObjectIsGzipEncoded
from gslib.utils.parallelism_framework_util import AtomicDict
from gslib.utils.parallelism_framework_util import CheckMultiprocessingAvailableAndInit
from gslib.utils.parallelism_framework_util import PutToQueueWithTimeout
from gslib.utils.posix_util import ATIME_ATTR
from gslib.utils.posix_util import ConvertDatetimeToPOSIX
from gslib.utils.posix_util import GID_ATTR
from gslib.utils.posix_util import MODE_ATTR
from gslib.utils.posix_util import MTIME_ATTR
from gslib.utils.posix_util import ParseAndSetPOSIXAttributes
from gslib.utils.posix_util import UID_ATTR
from gslib.utils.system_util import CheckFreeSpace
from gslib.utils.system_util import GetFileSize
from gslib.utils.system_util import GetStreamFromFileUrl
from gslib.utils.system_util import IS_WINDOWS
from gslib.utils.translation_helper import AddS3MarkerAclToObjectMetadata
from gslib.utils.translation_helper import CopyObjectMetadata
from gslib.utils.translation_helper import DEFAULT_CONTENT_TYPE
from gslib.utils.translation_helper import ObjectMetadataFromHeaders
from gslib.utils.translation_helper import PreconditionsFromHeaders
from gslib.utils.translation_helper import S3MarkerAclFromObjectMetadata
from gslib.utils.unit_util import DivideAndCeil
from gslib.utils.unit_util import HumanReadableToBytes
from gslib.utils.unit_util import MakeHumanReadable
from gslib.utils.unit_util import SECONDS_PER_DAY
from gslib.utils.unit_util import TEN_MIB
from gslib.wildcard_iterator import CreateWildcardIterator
def _CopyObjToObjDaisyChainMode(src_url, src_obj_metadata, dst_url, dst_obj_metadata, preconditions, gsutil_api, logger, decryption_key=None):
    """Copies from src_url to dst_url in "daisy chain" mode.

  See -D OPTION documentation about what daisy chain mode is.

  Args:
    src_url: Source CloudUrl
    src_obj_metadata: Metadata from source object
    dst_url: Destination CloudUrl
    dst_obj_metadata: Object-specific metadata that should be overidden during
                      the copy.
    preconditions: Preconditions to use for the copy.
    gsutil_api: gsutil Cloud API to use for the copy.
    logger: For outputting log messages.
    decryption_key: Base64-encoded decryption key for the source object, if any.

  Returns:
    (elapsed_time, bytes_transferred, dst_url with generation,
    md5 hash of destination) excluding overhead like initial GET.

  Raises:
    CommandException: if errors encountered.
  """
    if global_copy_helper_opts.preserve_acl and src_url.scheme != dst_url.scheme:
        raise NotImplementedError('Cross-provider cp -p not supported')
    if not global_copy_helper_opts.preserve_acl:
        dst_obj_metadata.acl = []
    progress_callback = None
    if global_copy_helper_opts.test_callback_file:
        with open(global_copy_helper_opts.test_callback_file, 'rb') as test_fp:
            progress_callback = pickle.loads(test_fp.read()).call
    compressed_encoding = ObjectIsGzipEncoded(src_obj_metadata)
    encryption_keywrapper = GetEncryptionKeyWrapper(config)
    start_time = time.time()
    upload_fp = DaisyChainWrapper(src_url, src_obj_metadata.size, gsutil_api, compressed_encoding=compressed_encoding, progress_callback=progress_callback, decryption_key=decryption_key)
    uploaded_object = None
    if src_obj_metadata.size == 0:
        uploaded_object = gsutil_api.UploadObject(upload_fp, object_metadata=dst_obj_metadata, canned_acl=global_copy_helper_opts.canned_acl, preconditions=preconditions, provider=dst_url.scheme, fields=UPLOAD_RETURN_FIELDS, size=src_obj_metadata.size, encryption_tuple=encryption_keywrapper)
    else:
        uploaded_object = gsutil_api.UploadObjectResumable(upload_fp, object_metadata=dst_obj_metadata, canned_acl=global_copy_helper_opts.canned_acl, preconditions=preconditions, provider=dst_url.scheme, fields=UPLOAD_RETURN_FIELDS, size=src_obj_metadata.size, progress_callback=FileProgressCallbackHandler(gsutil_api.status_queue, src_url=src_url, dst_url=dst_url, operation_name='Uploading').call, tracker_callback=_DummyTrackerCallback, encryption_tuple=encryption_keywrapper)
    end_time = time.time()
    try:
        _CheckCloudHashes(logger, src_url, dst_url, src_obj_metadata, uploaded_object)
    except HashMismatchException:
        if _RENAME_ON_HASH_MISMATCH:
            corrupted_obj_metadata = apitools_messages.Object(name=dst_obj_metadata.name, bucket=dst_obj_metadata.bucket, etag=uploaded_object.etag)
            dst_obj_metadata.name = dst_url.object_name + _RENAME_ON_HASH_MISMATCH_SUFFIX
            decryption_keywrapper = CryptoKeyWrapperFromKey(decryption_key)
            gsutil_api.CopyObject(corrupted_obj_metadata, dst_obj_metadata, provider=dst_url.scheme, decryption_tuple=decryption_keywrapper, encryption_tuple=encryption_keywrapper)
        gsutil_api.DeleteObject(dst_url.bucket_name, dst_url.object_name, generation=uploaded_object.generation, provider=dst_url.scheme)
        raise
    result_url = dst_url.Clone()
    result_url.generation = GenerationFromUrlAndString(result_url, uploaded_object.generation)
    PutToQueueWithTimeout(gsutil_api.status_queue, FileMessage(src_url, dst_url, end_time, message_type=FileMessage.FILE_DAISY_COPY, size=src_obj_metadata.size, finished=True))
    return (end_time - start_time, src_obj_metadata.size, result_url, uploaded_object.md5Hash)