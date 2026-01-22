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
def _ValidateAndCompleteDownload(logger, src_url, src_obj_metadata, dst_url, need_to_unzip, server_gzip, digesters, hash_algs, temporary_file_name, api_selector, bytes_transferred, gsutil_api, is_rsync=False, preserve_posix=False, use_stet=False):
    """Validates and performs necessary operations on a downloaded file.

  Validates the integrity of the downloaded file using hash_algs. If the file
  was compressed (temporarily), the file will be decompressed. Then, if the
  integrity of the file was successfully validated, the file will be moved
  from its temporary download location to its permanent location on disk.

  Args:
    logger: For outputting log messages.
    src_url: StorageUrl for the source object.
    src_obj_metadata: Metadata for the source object, potentially containing
                      hash values.
    dst_url: StorageUrl describing the destination file.
    need_to_unzip: If true, a temporary zip file was used and must be
                   uncompressed as part of validation.
    server_gzip: If true, the server gzipped the bytes (regardless of whether
                 the object metadata claimed it was gzipped).
    digesters: dict of {string, hash digester} that contains up-to-date digests
               computed during the download. If a digester for a particular
               algorithm is None, an up-to-date digest is not available and the
               hash must be recomputed from the local file.
    hash_algs: dict of {string, hash algorithm} that can be used if digesters
               don't have up-to-date digests.
    temporary_file_name: Temporary file name that was used for download.
    api_selector: The Cloud API implementation used (used tracker file naming).
    bytes_transferred: Number of bytes downloaded (used for logging).
    gsutil_api: Cloud API to use for service and status.
    is_rsync: Whether or not the caller is the rsync function. Used to determine
              if timeCreated should be used.
    preserve_posix: Whether or not to preserve the posix attributes.
    use_stet: If True, attempt to decrypt downloaded files with the STET
              binary if it's present on the system.

  Returns:
    An MD5 of the local file, if one was calculated as part of the integrity
    check.
  """
    final_file_name = dst_url.object_name
    digesters_succeeded = True
    for alg in digesters:
        if not digesters[alg]:
            digesters_succeeded = False
            break
    if digesters_succeeded:
        local_hashes = _CreateDigestsFromDigesters(digesters)
    else:
        local_hashes = _CreateDigestsFromLocalFile(gsutil_api.status_queue, hash_algs, temporary_file_name, src_url, src_obj_metadata)
    digest_verified = True
    hash_invalid_exception = None
    try:
        _CheckHashes(logger, src_url, src_obj_metadata, final_file_name, local_hashes)
        DeleteDownloadTrackerFiles(dst_url, api_selector)
    except HashMismatchException as e:
        if server_gzip:
            logger.debug('Hash did not match but server gzipped the content, will recalculate.')
            digest_verified = False
        elif api_selector == ApiSelector.XML:
            logger.debug('Hash did not match but server may have gzipped the content, will recalculate.')
            hash_invalid_exception = e
            digest_verified = False
        else:
            DeleteDownloadTrackerFiles(dst_url, api_selector)
            if _RENAME_ON_HASH_MISMATCH:
                os.rename(temporary_file_name, final_file_name + _RENAME_ON_HASH_MISMATCH_SUFFIX)
            else:
                os.unlink(temporary_file_name)
            raise
    if not (need_to_unzip or server_gzip):
        unzipped_temporary_file_name = temporary_file_name
    else:
        unzipped_temporary_file_name = temporary_file_util.GetTempFileName(dst_url)
        if bytes_transferred > TEN_MIB:
            logger.info('Uncompressing temporarily gzipped file to %s...', final_file_name)
        gzip_fp = None
        try:
            gzip_fp = gzip.open(temporary_file_name, 'rb')
            with open(unzipped_temporary_file_name, 'wb') as f_out:
                data = gzip_fp.read(GZIP_CHUNK_SIZE)
                while data:
                    f_out.write(data)
                    data = gzip_fp.read(GZIP_CHUNK_SIZE)
        except IOError as e:
            if 'Not a gzipped file' in str(e) and hash_invalid_exception:
                raise hash_invalid_exception
        finally:
            if gzip_fp:
                gzip_fp.close()
        os.unlink(temporary_file_name)
    if not digest_verified:
        try:
            local_hashes = _CreateDigestsFromLocalFile(gsutil_api.status_queue, hash_algs, unzipped_temporary_file_name, src_url, src_obj_metadata)
            _CheckHashes(logger, src_url, src_obj_metadata, final_file_name, local_hashes)
            DeleteDownloadTrackerFiles(dst_url, api_selector)
        except HashMismatchException:
            DeleteDownloadTrackerFiles(dst_url, api_selector)
            if _RENAME_ON_HASH_MISMATCH:
                os.rename(unzipped_temporary_file_name, unzipped_temporary_file_name + _RENAME_ON_HASH_MISMATCH_SUFFIX)
            else:
                os.unlink(unzipped_temporary_file_name)
            raise
    if use_stet:
        stet_util.decrypt_download(src_url, dst_url, unzipped_temporary_file_name, logger)
    os.rename(unzipped_temporary_file_name, final_file_name)
    ParseAndSetPOSIXAttributes(final_file_name, src_obj_metadata, is_rsync=is_rsync, preserve_posix=preserve_posix)
    if 'md5' in local_hashes:
        return local_hashes['md5']